#include <stdio.h>
#include "sndfile.h"
#include <cassert>
#include <math.h>
#include <cstring>

#define BUFFER_LEN 1024
#define MAX_CHANNELS 6 
#define ERROR_OPEN_INPUT -1
#define ERROR_OPEN_OUTPUT -2
#define EXTRA_CHANNELS -3
#define LACK_OF_FILES -4

SNDFILE* SIGNAL;
SNDFILE* WRITE;
SNDFILE* PREDICTION;
SF_INFO SFINFO_WRITE;
SF_INFO SFINFO_SIGNAL;
SF_INFO SFINFO_PREDICTION;

int open_sf_read(char* file, SNDFILE** sf_file, SF_INFO sf_file_info);
int open_sf_write(char* file, SNDFILE** sf_file, SF_INFO sf_file_info);
void process_data(SNDFILE* input, SNDFILE* pred);

int main(int argc, char* argv[])
{
    SIGNAL = NULL;
    PREDICTION = NULL;
    memset(&SFINFO_PREDICTION, 0, sizeof(SFINFO_PREDICTION));
    memset(&SFINFO_SIGNAL, 0, sizeof(SFINFO_SIGNAL));

    if(argc > 2)
    {
        open_sf_read(argv[1], &SIGNAL, SFINFO_SIGNAL);
        open_sf_read(argv[2], &PREDICTION, SFINFO_PREDICTION);

        sf_command(SIGNAL, SFC_SET_SCALE_FLOAT_INT_READ, NULL, SF_TRUE);
        sf_command(PREDICTION, SFC_SET_SCALE_FLOAT_INT_READ, NULL, SF_TRUE);

        process_data(SIGNAL, PREDICTION);

        sf_close(SIGNAL);
        sf_close(PREDICTION);

        return 0;
    }
    else
    {
        printf("Enter Signal file and Prediction file!\n");
        return LACK_OF_FILES;
    }
}

int open_sf_read(char* file, SNDFILE** sf_file, SF_INFO sf_file_info)
{
    if (!(*sf_file = sf_open(file, SFM_READ, &sf_file_info)))
    {
        printf("%s\n", sf_strerror(*sf_file));
        printf("Unable to open the input file\n");
        return ERROR_OPEN_INPUT;
    }

    if (sf_file_info.channels > MAX_CHANNELS)
    {
        printf("Not able to process more than %d channels\n", MAX_CHANNELS);
        return EXTRA_CHANNELS;
    }
}

int open_sf_write(char* file, SNDFILE** sf_file, SF_INFO sf_file_info)
{
    memset(&sf_file_info, 0, sizeof(sf_file_info));

    sf_file_info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    sf_file_info.channels = 2;
    sf_file_info.samplerate = 44100;

    if (!(*sf_file = sf_open(file, SFM_WRITE, &sf_file_info)))
    {
        printf("%s\n", sf_strerror(*sf_file));
        printf("Unable to open the input file\n");
        return ERROR_OPEN_INPUT;
    }
}

void process_data(SNDFILE* input, SNDFILE* pred)
{
    int readcount = 0;
    short data1 [BUFFER_LEN];
    short data2 [BUFFER_LEN];

     while ((readcount = sf_read_short(input, data1, BUFFER_LEN)) && (readcount = sf_read_short(pred, data2, BUFFER_LEN)))
    {  
        for (int i = 0; i < readcount; i++)
            printf("%hd ", (data1[i] - data2[i]));

        printf("\n");
    }
}