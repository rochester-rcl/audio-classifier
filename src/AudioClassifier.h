//
//  imageClassifier.h
//  imageClassifier
//
//  Created by JoshR on 2/10/16.
//  Copyright (c) 2016 joshr. All rights reserved.
// Compile using GCC: g++ main.cpp -o imageClassifier `pkg-config --cflags --libs opencv` -lboost_system -lboost_filesystem  -lboost_program_options -lboost_serialization
//

#ifndef imageClassifier_imageClassifier_h
#define imageClassifier_imageClassifier_h
#endif

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/ml/ml.hpp>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <iterator>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include "sox.h"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace ml;
using namespace boost::filesystem;
using namespace boost::program_options;
using namespace boost::serialization;
using namespace boost::archive;

extern "C" {

  const char* generateSpectrogram(sox_format_t *inputInfo, char * in, size_t length, string spectrogramPath) {
    size_t bufSize = length * sizeof(sox_sample_t);
    sox_encodinginfo_t rawEncoding;
    sox_encodinginfo_t inputEncoding = inputInfo->encoding;

    rawEncoding.encoding = SOX_ENCODING_SIGN2;
    rawEncoding.bits_per_sample = inputEncoding.bits_per_sample;

    sox_format_t *inCopy;
    assert(inCopy = sox_open_mem_read(in, bufSize, &inputInfo->signal, &rawEncoding, NULL));
    sox_format_t *out;
    char *outbuf;
    size_t outbuf_size;
    char *args[10];

    // Need to write the output to a dummy buffer b/c
    assert(out = sox_open_memstream_write(&outbuf, &outbuf_size, &inCopy->signal, &inCopy->encoding, "raw", NULL));
    sox_effects_chain_t *chain = sox_create_effects_chain(&inCopy->encoding, &inCopy->encoding);
    // Input effect
    sox_effect_t *effect;
    effect = sox_create_effect(sox_find_effect("input"));
    args[0] = (char *)inCopy;
    assert(sox_effect_options(effect, 1, args) == SOX_SUCCESS);
    assert(sox_add_effect(chain, effect, &inCopy->signal, &inCopy->signal) == SOX_SUCCESS);
    free(effect);

    // Spectrogram
    effect = sox_create_effect(sox_find_effect("spectrogram"));
    spectrogramPath.insert(0,"-o");
    args[0] = const_cast<char *>("-r");
    args[1] = const_cast<char *>(spectrogramPath.c_str());
    assert(sox_effect_options(effect, 2, args) == SOX_SUCCESS);
    assert(sox_add_effect(chain, effect, &inCopy->signal, &inCopy->signal) == SOX_SUCCESS);
    free(effect);

    // Output effect
    effect = sox_create_effect(sox_find_effect("output"));
    args[0] = (char *)out;
    assert(sox_effect_options(effect, 1, args) == SOX_SUCCESS);
    assert(sox_add_effect(chain, effect, &inCopy->signal, &inCopy->signal) == SOX_SUCCESS);
    free(effect);
    sox_flow_effects(chain, NULL, NULL);
    sox_delete_effects_chain(chain);
    sox_close(out);
    sox_close(inCopy);

    const char* message = "temp_spectrogram.png";
    return message;
  }

  bool initSox() {
    if (sox_init() == SOX_SUCCESS && sox_format_init() == SOX_SUCCESS) {
      return true;
    }
    return false;
  }

  sox_format_t *openAudio(const char* filePath) {
    static sox_format_t *in;
    assert(in = sox_open_read(filePath, NULL, NULL, NULL));
    return in;
  }

  void readAudio(sox_format_t *input, char * buffer, size_t bufferSize, size_t len) {
    sox_format_t *out;
    sox_sample_t samples[len];
    sox_signalinfo_t outInfo;
    outInfo.length = len;
    outInfo.channels = input->signal.channels;
    outInfo.rate = input->signal.rate;
    assert(out = sox_open_mem_write(buffer, bufferSize, &outInfo, NULL, "sox", NULL));
    size_t samplesRead;
    samplesRead = sox_read(input, samples, len);
    assert(sox_write(out, samples, samplesRead) == samplesRead);
  }

}

class VocabBuilder {


    public:

        //properties
        string vocabOutputFile;
        string inputVideoPath;
        string svmLabels;
        string trainFilesDirectory;
        Mat origVocab;
        vector<string> imageFiles;


        //Member functions

        //Constructors
        VocabBuilder(string directory, string outputFile); // Overloading constructors
        VocabBuilder(string directory);
        VocabBuilder(string directory, FileStorage vocab);
        VocabBuilder(string labelsFile, FileStorage vocab, string outputFile, string inputFile);

        //Static
        static string getClassFromPath(string filePath);
        static vector<string> getTestFiles(string filePath);
        static void getSpectrogram(string filePath);

        //Non-static
        void listFiles(string directory);
        void createVocab();
        void vocab2File();
        void SVM2File();
        void testSVM();
        void createTrainingData();


};
