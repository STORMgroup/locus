#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <zlib.h>

#include "htslib/kseq.h"

KSEQ_INIT(gzFile, gzread)

#define DEFAULT_WINDOW_LENGTH 4096
#define REFERENCE_SHARD_BASE_COUNT (4096ULL * 10000ULL)
#define ACCEPTED_LOG_INTERVAL 10000
#define SPLIT_PATH_CAPACITY 4096
#define SPLIT_NAME_CAPACITY 512

typedef struct {
    const char *output_dir;
    const char *dataset_id;
    unsigned long long window_length;
    unsigned long long windows_per_shard;
    double max_repeat_fraction;
    unsigned long long repeat_filtered_count;
    gzFile shard;
    gzFile read_manifest;
    gzFile shard_manifest;
    gzFile rejected_report;
    gzFile discarded_report;
    char shard_name[SPLIT_NAME_CAPACITY];
    char first_read_id[SPLIT_NAME_CAPACITY];
    char last_read_id[SPLIT_NAME_CAPACITY];
    unsigned long long shard_index;
    unsigned long long records_in_shard;
    unsigned long long output_record_count;
    unsigned long long retained_base_count;
    unsigned long long retained_n_count;
    unsigned long long reads_with_n_count;
    unsigned long long skipped_all_n_read_count;
    unsigned long long skipped_all_n_base_count;
    unsigned long long discarded_tail_count;
    unsigned long long discarded_base_count;
    unsigned long long rejected_base_count;
    unsigned long long rejected_n_count;
} split_writer;

static void report_accepted_progress(unsigned long long record_count,
                                     unsigned long long base_count) {
    if (record_count % ACCEPTED_LOG_INTERVAL == 0) {
        fprintf(stderr,
                "Accepted %llu record(s), totaling %llu base(s), so far\n",
                record_count, base_count);
    }
}

static int is_valid_base(char base) {
    /* N and every other symbol end the current window, so no window contains or spans an N. */
    return base == 'A' || base == 'C' || base == 'G' || base == 'T' ||
           base == 'a' || base == 'c' || base == 'g' || base == 't';
}

static int make_directory(const char *path) {
    if (mkdir(path, 0775) == 0) {
        return 0;
    }
    if (errno == EEXIST) {
        struct stat info;
        if (stat(path, &info) == 0 && S_ISDIR(info.st_mode)) {
            return 0;
        }
    }
    fprintf(stderr, "Could not create directory %s: %s\n", path,
            strerror(errno));
    return -1;
}

static int make_output_path(char *destination, size_t destination_size,
                            const char *directory, const char *relative_path) {
    int length = snprintf(destination, destination_size, "%s/%s", directory,
                          relative_path);
    if (length < 0 || (size_t)length >= destination_size) {
        fprintf(stderr, "Output path is too long: %s/%s\n", directory,
                relative_path);
        return -1;
    }
    return 0;
}

static int valid_dataset_id(const char *dataset_id) {
    if (*dataset_id == '\0') {
        return 0;
    }
    for (const unsigned char *cursor = (const unsigned char *)dataset_id;
         *cursor != '\0'; ++cursor) {
        if (!isalnum(*cursor) && *cursor != '.' && *cursor != '_' &&
            *cursor != '-') {
            return 0;
        }
    }
    return 1;
}

static void sanitize_identifier(const char *source, char *destination,
                                size_t destination_size) {
    size_t output_index = 0;
    while (*source != '\0' && output_index + 1 < destination_size) {
        unsigned char character = (unsigned char)*source++;
        destination[output_index++] =
            (isalnum(character) || character == '.' || character == '_' ||
             character == '-')
                ? (char)character
                : '_';
    }
    destination[output_index] = '\0';
}

static int write_wrapped_sequence(gzFile output, const char *sequence,
                                  size_t sequence_length) {
    for (size_t offset = 0; offset < sequence_length; offset += 80) {
        size_t line_length = sequence_length - offset;
        if (line_length > 80) {
            line_length = 80;
        }
        if (gzwrite(output, sequence + offset, (unsigned int)line_length) !=
                (int)line_length ||
            gzputc(output, '\n') == -1) {
            return -1;
        }
    }
    return 0;
}

static int open_next_shard(split_writer *writer) {
    char relative_path[SPLIT_NAME_CAPACITY];
    char full_path[SPLIT_PATH_CAPACITY];
    int length = snprintf(relative_path, sizeof(relative_path),
                          "shards/%s-w%llu-shard-%05llu.fasta.gz",
                          writer->dataset_id, writer->window_length,
                          writer->shard_index);
    if (length < 0 || (size_t)length >= sizeof(relative_path) ||
        make_output_path(full_path, sizeof(full_path), writer->output_dir,
                         relative_path) != 0) {
        return -1;
    }

    writer->shard = gzopen(full_path, "wb");
    if (writer->shard == NULL) {
        fprintf(stderr, "Could not open shard %s for writing\n", full_path);
        return -1;
    }
    memcpy(writer->shard_name, relative_path, (size_t)length + 1);
    writer->records_in_shard = 0;
    writer->first_read_id[0] = '\0';
    writer->last_read_id[0] = '\0';
    return 0;
}

static int close_current_shard(split_writer *writer) {
    if (writer->shard == NULL) {
        return 0;
    }

    int status = 0;
    if (gzclose(writer->shard) != Z_OK) {
        fprintf(stderr, "Could not finish writing shard %s\n",
                writer->shard_name);
        status = -1;
    }
    writer->shard = NULL;

    if (status == 0 &&
        gzprintf(writer->shard_manifest, "%s\t%llu\t%llu\t%s\t%s\n",
                 writer->shard_name, writer->records_in_shard,
                 writer->records_in_shard * writer->window_length,
                 writer->first_read_id, writer->last_read_id) <= 0) {
        fprintf(stderr, "Could not write shard manifest row\n");
        status = -1;
    }
    ++writer->shard_index;
    return status;
}

static int write_split_record(split_writer *writer, const char *source_name,
                              unsigned long long source_record_ordinal,
                              size_t source_start, const char *sequence,
                              unsigned long long lowercase_count) {
    unsigned long long window_length = writer->window_length;
    unsigned long long n_count = 0;
    for (size_t position = 0; position < window_length; ++position) {
        if (sequence[position] == 'N') {
            ++n_count;
        }
    }
    if (n_count == window_length) {
        ++writer->skipped_all_n_read_count;
        writer->skipped_all_n_base_count += window_length;
        return 0;
    }
    if (writer->max_repeat_fraction < 1.0 &&
        (double)lowercase_count / (double)window_length >
            writer->max_repeat_fraction) {
        ++writer->repeat_filtered_count;
        return 0;
    }

    if (writer->shard == NULL && open_next_shard(writer) != 0) {
        return -1;
    }

    size_t source_end = source_start + (size_t)window_length - 1;
    char safe_source[SPLIT_NAME_CAPACITY];
    char read_id[SPLIT_NAME_CAPACITY];
    sanitize_identifier(source_name, safe_source, sizeof(safe_source));
    int id_length = snprintf(
        read_id, sizeof(read_id), "%s_r%06llu_%s_%012zu_%012zu",
        writer->dataset_id, source_record_ordinal, safe_source, source_start,
        source_end);
    if (id_length < 0 || (size_t)id_length >= sizeof(read_id)) {
        fprintf(stderr, "Read identifier is too long for source %s\n",
                source_name);
        return -1;
    }

    if (gzprintf(writer->shard,
                 ">%s source=%s start=%zu end=%zu strand=+\n", read_id,
                 source_name, source_start, source_end) <= 0 ||
        write_wrapped_sequence(writer->shard, sequence,
                              (size_t)window_length) != 0) {
        fprintf(stderr, "Could not write read %s\n", read_id);
        return -1;
    }

    unsigned long long record_in_shard = writer->records_in_shard + 1;
    if (gzprintf(writer->read_manifest,
                 "%s\t%s\t%llu\t%zu\t%zu\t%llu\t%llu\t+\t%s\t%llu\n",
                 read_id, source_name, source_record_ordinal, source_start,
                 source_end, window_length, n_count, writer->shard_name,
                 record_in_shard) <= 0) {
        fprintf(stderr, "Could not write read manifest row for %s\n", read_id);
        return -1;
    }

    if (writer->records_in_shard == 0) {
        memcpy(writer->first_read_id, read_id, (size_t)id_length + 1);
    }
    memcpy(writer->last_read_id, read_id, (size_t)id_length + 1);
    ++writer->records_in_shard;
    ++writer->output_record_count;
    writer->retained_base_count += window_length;
    writer->retained_n_count += n_count;
    writer->reads_with_n_count += n_count > 0;
    report_accepted_progress(writer->output_record_count,
                             writer->retained_base_count);

    if (writer->records_in_shard == writer->windows_per_shard &&
        close_current_shard(writer) != 0) {
        return -1;
    }
    return 0;
}

static int report_discarded_tail(split_writer *writer, const char *source_name,
                                 unsigned long long source_record_ordinal,
                                 size_t source_start, size_t sequence_length,
                                 const char *reason) {
    if (sequence_length == 0) {
        return 0;
    }
    size_t source_end = source_start + sequence_length - 1;
    if (gzprintf(writer->discarded_report,
                 "%s\t%llu\t%zu\t%zu\t%zu\t%s\n", source_name,
                 source_record_ordinal, source_start, source_end,
                 sequence_length, reason) <= 0) {
        fprintf(stderr, "Could not write discarded-tail report row\n");
        return -1;
    }
    ++writer->discarded_tail_count;
    writer->discarded_base_count += sequence_length;
    return 0;
}

static int write_dataset_summary(const split_writer *writer,
                                 const char *source_path,
                                 unsigned long long source_record_count) {
    char summary_path[SPLIT_PATH_CAPACITY];
    if (make_output_path(summary_path, sizeof(summary_path), writer->output_dir,
                         "dataset.json") != 0) {
        return -1;
    }
    FILE *summary = fopen(summary_path, "w");
    if (summary == NULL) {
        fprintf(stderr, "Could not open %s for writing\n", summary_path);
        return -1;
    }
    int write_status = fprintf(
        summary,
        "{\n"
        "  \"dataset_id\": \"%s\",\n"
        "  \"source\": \"%s\",\n"
        "  \"window_length\": %llu,\n"
        "  \"stride\": %llu,\n"
        "  \"windows_per_shard\": %llu,\n"
        "  \"source_record_count\": %llu,\n"
        "  \"shard_count\": %llu,\n"
        "  \"read_count\": %llu,\n"
        "  \"retained_base_count\": %llu,\n"
        "  \"retained_n_count\": %llu,\n"
        "  \"reads_with_n_count\": %llu,\n"
        "  \"reads_with_n_fraction\": %.10f,\n"
        "  \"skipped_all_n_read_count\": %llu,\n"
        "  \"skipped_all_n_base_count\": %llu,\n"
        "  \"discarded_tail_count\": %llu,\n"
        "  \"discarded_tail_base_count\": %llu,\n"
        "  \"rejected_base_count\": %llu,\n"
        "  \"rejected_n_count\": %llu,\n"
        "  \"max_repeat_fraction\": %.4f,\n"
        "  \"repeat_filtered_count\": %llu\n"
        "}\n",
        writer->dataset_id, source_path, writer->window_length,
        writer->window_length,
        writer->windows_per_shard, source_record_count, writer->shard_index,
        writer->output_record_count, writer->retained_base_count,
        writer->retained_n_count, writer->reads_with_n_count,
        writer->output_record_count == 0
            ? 0.0
            : (double)writer->reads_with_n_count /
                  (double)writer->output_record_count,
        writer->skipped_all_n_read_count,
        writer->skipped_all_n_base_count,
        writer->discarded_tail_count, writer->discarded_base_count,
        writer->rejected_base_count, writer->rejected_n_count,
        writer->max_repeat_fraction, writer->repeat_filtered_count);
    if (write_status < 0 || fclose(summary) != 0) {
        fprintf(stderr, "Could not finish writing %s\n", summary_path);
        return -1;
    }
    return 0;
}

static int split_fasta(const char *path, const char *output_dir,
                       const char *dataset_id,
                       unsigned long long window_length,
                       double max_repeat_fraction) {
    int status = 1;
    if (!valid_dataset_id(dataset_id)) {
        fprintf(stderr,
                "Dataset ID must contain only letters, digits, '.', '_', or '-'.\n");
        return 1;
    }
    if (window_length == 0 || window_length > 1000000ULL) {
        fprintf(stderr, "Window length must be a positive number of bases (<= 1,000,000).\n");
        return 1;
    }

    char directory_path[SPLIT_PATH_CAPACITY];
    if (make_directory(output_dir) != 0) {
        return 1;
    }
    const char *subdirectories[] = {"shards", "manifests", "reports"};
    for (size_t i = 0; i < sizeof(subdirectories) / sizeof(subdirectories[0]);
         ++i) {
        if (make_output_path(directory_path, sizeof(directory_path), output_dir,
                             subdirectories[i]) != 0 ||
            make_directory(directory_path) != 0) {
            return 1;
        }
    }

    gzFile input = gzopen(path, "rb");
    if (input == NULL) {
        fprintf(stderr, "Could not open %s\n", path);
        return 1;
    }

    kseq_t *record = kseq_init(input);
    if (record == NULL) {
        fprintf(stderr, "Could not initialize the FASTA/FASTQ reader.\n");
        gzclose(input);
        return 1;
    }

    split_writer writer = {0};
    writer.output_dir = output_dir;
    writer.dataset_id = dataset_id;
    writer.window_length = window_length;
    writer.max_repeat_fraction = max_repeat_fraction;
    writer.windows_per_shard = REFERENCE_SHARD_BASE_COUNT / window_length;
    if (writer.windows_per_shard == 0) {
        writer.windows_per_shard = 1;
    }
    unsigned long long source_record_count = 0;

    char *chunk = malloc((size_t)window_length);
    if (chunk == NULL) {
        fprintf(stderr, "Could not allocate a %llu-byte read buffer\n",
                window_length);
        gzclose(input);
        kseq_destroy(record);
        return 1;
    }

    struct {
        const char *relative_path;
        gzFile *file;
    } gzip_outputs[] = {
        {"manifests/reads.tsv.gz", &writer.read_manifest},
        {"manifests/shards.tsv.gz", &writer.shard_manifest},
        {"reports/rejected_bases.tsv.gz", &writer.rejected_report},
        {"reports/discarded_tails.tsv.gz", &writer.discarded_report},
    };
    for (size_t i = 0; i < sizeof(gzip_outputs) / sizeof(gzip_outputs[0]); ++i) {
        char output_path[SPLIT_PATH_CAPACITY];
        if (make_output_path(output_path, sizeof(output_path), output_dir,
                             gzip_outputs[i].relative_path) != 0) {
            goto cleanup;
        }
        *gzip_outputs[i].file = gzopen(output_path, "wb");
        if (*gzip_outputs[i].file == NULL) {
            fprintf(stderr, "Could not open %s for writing\n", output_path);
            goto cleanup;
        }
    }

    if (gzprintf(writer.read_manifest,
                 "read_id\tsource\tsource_record_ordinal\tstart_1based\t"
                 "end_1based\tlength\tn_count\tstrand\tshard\trecord_in_shard\n") <= 0 ||
        gzprintf(writer.shard_manifest,
                 "shard\tread_count\tbase_count\tfirst_read_id\tlast_read_id\n") <= 0 ||
        gzprintf(writer.rejected_report,
                 "source\tsource_record_ordinal\tposition_1based\tbase\n") <= 0 ||
        gzprintf(writer.discarded_report,
                 "source\tsource_record_ordinal\tstart_1based\tend_1based\t"
                 "length\treason\n") <= 0) {
        fprintf(stderr, "Could not write output manifest/report headers\n");
        goto cleanup;
    }

    size_t chunk_length = 0;
    size_t chunk_start = 0;
    unsigned long long chunk_lowercase_count = 0;
    int read_status;
    while ((read_status = kseq_read(record)) >= 0) {
        ++source_record_count;
        if (record->seq.l == 0) {
            fprintf(stderr, "Skipping empty record %s\n", record->name.s);
            continue;
        }

        chunk_length = 0;
        chunk_start = 0;
        chunk_lowercase_count = 0;
        for (size_t position = 0; position < record->seq.l; ++position) {
            char base = record->seq.s[position];
            if (is_valid_base(base)) {
                if (chunk_length == 0) {
                    chunk_start = position + 1;
                }
                if (base >= 'a' && base <= 'z') {
                    ++chunk_lowercase_count;
                    base = (char)(base - 'a' + 'A');
                }
                chunk[chunk_length++] = base;

                if (chunk_length == window_length) {
                    if (write_split_record(&writer, record->name.s,
                                           source_record_count, chunk_start,
                                           chunk, chunk_lowercase_count) != 0) {
                        goto cleanup;
                    }
                    chunk_length = 0;
                    chunk_start = 0;
                    chunk_lowercase_count = 0;
                }
                continue;
            }

            if (report_discarded_tail(&writer, record->name.s,
                                      source_record_count, chunk_start,
                                      chunk_length, "before_invalid") != 0) {
                goto cleanup;
            }
            chunk_length = 0;
            chunk_start = 0;
            chunk_lowercase_count = 0;

            ++writer.rejected_base_count;
            if (base == 'N' || base == 'n') {
                ++writer.rejected_n_count;
            }
            if (gzprintf(writer.rejected_report, "%s\t%llu\t%zu\t%c\n",
                         record->name.s, source_record_count, position + 1,
                         base) <= 0) {
                fprintf(stderr, "Could not write rejected base report\n");
                goto cleanup;
            }

            if (writer.rejected_base_count % 1000000 == 0) {
                fprintf(stderr, "Reported %llu rejected base(s) so far\n",
                        writer.rejected_base_count);
            }
        }

        if (report_discarded_tail(&writer, record->name.s,
                                  source_record_count, chunk_start,
                                  chunk_length, "end_of_record") != 0) {
            goto cleanup;
        }
        chunk_length = 0;
        chunk_start = 0;
        chunk_lowercase_count = 0;
    }

    if (read_status < -1) {
        fprintf(stderr, "Error while reading %s (kseq error %d)\n",
                path, read_status);
        goto cleanup;
    }

    if (writer.output_record_count == 0) {
        fprintf(stderr,
                "No complete %llu-base ACGT windows were found in %s\n",
                window_length, path);
        goto cleanup;
    }
    status = 0;

cleanup:
    free(chunk);
    if (close_current_shard(&writer) != 0) {
        status = 1;
    }
    gzFile *files_to_close[] = {
        &writer.read_manifest,
        &writer.shard_manifest,
        &writer.rejected_report,
        &writer.discarded_report,
    };
    for (size_t i = 0; i < sizeof(files_to_close) / sizeof(files_to_close[0]);
         ++i) {
        if (*files_to_close[i] != NULL) {
            if (gzclose(*files_to_close[i]) != Z_OK) {
                fprintf(stderr, "Could not finish writing a dataset metadata file\n");
                status = 1;
            }
            *files_to_close[i] = NULL;
        }
    }
    kseq_destroy(record);
    if (gzclose(input) != Z_OK) {
        fprintf(stderr, "Could not finish reading %s\n", path);
        status = 1;
    }

    if (status == 0 &&
        write_dataset_summary(&writer, path, source_record_count) != 0) {
        status = 1;
    }
    if (status == 0) {
        fprintf(stdout, "Read %llu source FASTA record(s)\n",
                source_record_count);
        fprintf(stdout,
                "Wrote %llu exact %llu-base ACGT read(s) in %llu shard(s)\n",
                writer.output_record_count, window_length, writer.shard_index);
        fprintf(stdout, "Retained %llu base(s), including %llu N(s)\n",
                writer.retained_base_count, writer.retained_n_count);
        fprintf(stdout, "%llu read(s) contain N (%.6f%%)\n",
                writer.reads_with_n_count,
                writer.output_record_count == 0
                    ? 0.0
                    : 100.0 * (double)writer.reads_with_n_count /
                          (double)writer.output_record_count);
        fprintf(stdout, "Skipped %llu all-N read(s), totaling %llu bases\n",
                writer.skipped_all_n_read_count,
                writer.skipped_all_n_base_count);
        fprintf(stdout,
                "Discarded %llu ACGT base(s) across %llu incomplete tail(s)\n",
                writer.discarded_base_count, writer.discarded_tail_count);
        fprintf(stdout,
                "Reported %llu rejected base(s), including %llu N(s)\n",
                writer.rejected_base_count, writer.rejected_n_count);
        if (writer.max_repeat_fraction < 1.0) {
            fprintf(stdout,
                    "Repeat-filtered %llu window(s) (>%.1f%% soft-masked)\n",
                    writer.repeat_filtered_count,
                    100.0 * writer.max_repeat_fraction);
        }
    }
    return status;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        fprintf(stderr,
                "Usage: %s INPUT.fasta[.gz] OUTPUT_DIRECTORY DATASET_ID "
                "[WINDOW_LENGTH] [MAX_REPEAT_FRACTION]\n"
                "  WINDOW_LENGTH defaults to %d if omitted.\n"
                "  MAX_REPEAT_FRACTION (0.0-1.0) rejects a window whose "
                "fraction of soft-masked\n"
                "  (lowercase, i.e. repeat-annotated) bases exceeds this "
                "value; defaults to 1.0\n"
                "  (no filtering) if omitted.\n",
                argv[0], DEFAULT_WINDOW_LENGTH);
        return 1;
    }

    unsigned long long window_length = DEFAULT_WINDOW_LENGTH;
    if (argc >= 5) {
        char *end = NULL;
        window_length = strtoull(argv[4], &end, 10);
        if (end == argv[4] || *end != '\0') {
            fprintf(stderr, "WINDOW_LENGTH must be a positive integer, got: %s\n",
                    argv[4]);
            return 1;
        }
    }

    double max_repeat_fraction = 1.0;
    if (argc == 6) {
        char *end = NULL;
        max_repeat_fraction = strtod(argv[5], &end);
        if (end == argv[5] || *end != '\0' || max_repeat_fraction < 0.0 ||
            max_repeat_fraction > 1.0) {
            fprintf(stderr,
                    "MAX_REPEAT_FRACTION must be a number in [0.0, 1.0], got: %s\n",
                    argv[5]);
            return 1;
        }
    }

    return split_fasta(argv[1], argv[2], argv[3], window_length,
                       max_repeat_fraction);
}
