//
//  main.cpp
//  AudioClassifier
//
//  Created by JoshR on 2/10/16.
//  Copyright (c) 2016 joshr. All rights reserved.
//

#include "audio-classifier.cpp"

int main(int argc, const char* argv[]) {

    //Get command line options
    options_description description("Options");
    description.add_options()
    ("help,h", "Print help messages")
    ("train_dir,td", value<string>(), "Directory with training set")
    ("input,i", value<string>(), "Input video file to classify")
    ("vocab,v", value<string>(), "Input vocabulary file (yml)")
    ("output_vocab,ov", value<string>(), "Output vocabulary file")
    ("output,o", value<string>(), "Output directory")
    ("svm,s", value<string>(), "SVM file")
    ("labels,l", value<string>(), "Labels file");


    variables_map variables;
    store(parse_command_line(argc, argv, description), variables);
    try {
        if(variables.count("help")){
            cout << "Command line tool that classifies audio files based on an existing training set of spectrograms" << endl;
            cout << "If input video, output directory, and vocabulary file (yml) are specified, you can use the classification tool" << endl << endl;
            cout << "e.g. AudioClassifier --input(-i) --vocab(v) --output(-o)" << endl << endl;
            cout << "Otherwise, specify a training directory and an output name for the vocabulary file (must be YAML)" << endl << endl;
            cout << "e.g. AudioClassifier --train_dir(-td) --output_vocab(-ov)" << endl << endl;

            cout << "A final option is to do this all in one go, without the vocabulary file. Warning: It's pretty slow" << endl << endl;
            cout << "e.g AudioClassifier --train_dir(td) --input(-i) --output(-o)" << endl <<endl;

            return 0;
        }
        notify(variables);
    } catch (boost::program_options::error& e) {
        cerr << "ERROR: " << e.what() << endl;
        cerr << description << endl;
        return 1;
    }

    //create yaml file for vocabulary taken from train directory
    if (variables.count("train_dir") && variables.count("output_vocab")){

        string directoryPath = variables["train_dir"].as<string>();
        string vocabOutputFile = variables["output_vocab"].as<string>();
        VocabBuilder vocab(directoryPath, vocabOutputFile); // get files in provided directory, output YAML file
        vocab.createVocab();
        vocab.vocab2File();

    //create svm file from vocab yaml file and save to output
    } else if (variables.count("train_dir") && variables.count("vocab") && variables.count("output")){

        string directoryPath = variables["train_dir"].as<string>();
        string vocabFile = variables["vocab"].as<string>();
        string outputDir = variables["output"].as<string>();

        FileStorage inputVocab(vocabFile, FileStorage::READ);
        cout << "Reading " << vocabFile << endl;
        VocabBuilder vocab(directoryPath, inputVocab);
        vocab.createTrainingData();

    //create vocabulary and svm files
    } else if (variables.count("train_dir") && variables.count("output")) {

        string outputDir = variables["output"].as<string>();
        string directoryPath = variables["train_dir"].as<string>();
        VocabBuilder vocab(directoryPath);
        vocab.createVocab();
        vocab.createTrainingData();
        //vocab.testSVM();

    //Load in vocab yaml file and boost archive with svm path and labels. Run classifier on input video
    } else if (variables.count("labels") && variables.count("vocab") && variables.count("input") && variables.count("output")) {

        string inputFile = variables["input"].as<string>();
        string outputDir = variables["output"].as<string>();
        string labels = variables["labels"].as<string>();
        string vocabFile = variables["vocab"].as<string>();
        FileStorage inputVocab(vocabFile, FileStorage::READ);
        cout << "Reading " << vocabFile << endl;
        VocabBuilder vocab(labels, inputVocab, outputDir, inputFile);
        vocab.testSVM();

    }

    else {
        cout << "Incorrect arguments set. Please run audio-classifier --help for assistance" << endl;
        return 1;
    }


    return 0;


}
