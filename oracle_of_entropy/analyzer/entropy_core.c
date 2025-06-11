#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: ./oracle <input_file>\n");
        return 1;
    }

    FILE *fp = fopen(argv[1], "rb");
    if (!fp) {
        perror("File open failed");
        return 1;
    }

    int counts[256] = {0};
    int total = 0;
    int c;

    while ((c = fgetc(fp)) != EOF) {
        counts[(unsigned char)c]++;
        total++;
    }

    fclose(fp);

    if (total == 0) {
        printf("Empty file or read error.\n");
        return 1;
    }

    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (counts[i] == 0) continue;
        double p = (double)counts[i] / total;
        entropy -= p * log2(p);
    }

    printf("Entropy: %.4f bits/byte\n", entropy);
    return 0;
}

