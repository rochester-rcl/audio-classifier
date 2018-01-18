//
//  main.cpp
//  AudioClassifier
//
//  Created by JoshR on 2/10/16.
//  Copyright (c) 2016 joshr. All rights reserved.
//



#include "AudioClassifier.h"

//Member functions
//Constructors (overloaded)
VocabBuilder::VocabBuilder(string directory, string output) {

    trainFilesDirectory = directory;
    vocabOutputFile = output;
    listFiles(trainFilesDirectory);

}

VocabBuilder::VocabBuilder(string directory){

    trainFilesDirectory = directory;
    listFiles(trainFilesDirectory);
}

VocabBuilder::VocabBuilder(string directory, FileStorage vocabFile){

    trainFilesDirectory = directory;
    vocabFile["Vocabulary"] >> origVocab;
    listFiles(trainFilesDirectory);

}

VocabBuilder::VocabBuilder(string labelsFile, FileStorage vocabFile, string outputFile, string inputFile){

    vocabFile["Vocabulary"] >> origVocab;
    svmLabels = labelsFile;
    inputVideoPath = inputFile;

}

//Static member functions

void VocabBuilder::getSpectrogram(string filePath) {
  //const char* png = generateSpectrogram(filePath.c_str());
}

string VocabBuilder::getClassFromPath(string filePath){

    vector<string> tokenized;
    stringstream stream(filePath);
    char delimiter = '/';
    string token;
    string className;
    while (getline(stream, token, delimiter)){
        tokenized.push_back(token);
    }

    for (vector<int>::size_type i = 0; i != tokenized.size(); i++){

        if (i == tokenized.size() - 2) {
            className = tokenized[i];
        }

    }

    return className;

}

vector<string> VocabBuilder::getTestFiles(string filepath){

    static const string extensionsArray[] = {".tif", ".jpg", ".png"};

    vector<string> extensions (extensionsArray, extensionsArray + sizeof(extensionsArray) / sizeof(extensionsArray[0]));

    vector<string> fileNames;
    path inputDirectory = filepath;
    recursive_directory_iterator iter((inputDirectory)), endOfDirectory;

    for (recursive_directory_iterator iterator(inputDirectory); iterator != endOfDirectory; ++iterator)
    {
        if (is_regular_file(iterator->path())) {

            string currentFile = iterator->path().string();
            string extension = iterator->path().extension().string();
            cout << "Scanning " << currentFile << "\n";

            if (find(extensions.begin(), extensions.end(), extension) != extensions.end()){

                fileNames.push_back(currentFile);

            }
        }
    }
    return fileNames;
}

//Non-static

void VocabBuilder::listFiles(string argument){

    static const string extensionsArray[] = {".tif", ".jpg", ".png"};

    vector<string> extensions (extensionsArray, extensionsArray + sizeof(extensionsArray) / sizeof(extensionsArray[0]));

    vector<string> fileNames;
    path inputDirectory = argument;
    recursive_directory_iterator iter((inputDirectory)), endOfDirectory;

    for (recursive_directory_iterator iterator(inputDirectory); iterator != endOfDirectory; ++iterator)
    {
        if (is_regular_file(iterator->path())) {

            string currentFile = iterator->path().string();
            string extension = iterator->path().extension().string();
            cout << "Scanning " << currentFile << "\n";

            if (find(extensions.begin(), extensions.end(), extension) != extensions.end()){

                fileNames.push_back(currentFile);

            }
        }
    }
    imageFiles = fileNames;
}

void VocabBuilder::createVocab() {
    Ptr<SurfDescriptorExtractor> extractor = SurfDescriptorExtractor::create();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    Mat trainingDescriptors(1, extractor->descriptorSize(), extractor->descriptorType());
    Ptr<SURF> detector = SURF::create(400);

    cout << "Building vocabulary.\n";

    for (vector<int>::size_type i = 0; i != imageFiles.size(); i++){
        Mat img = imread(imageFiles[i]);
        cout << "Reading image " << i << "\n";
        detector->detect(img, keypoints);
        extractor->compute(img, keypoints, descriptors);
        trainingDescriptors.push_back(descriptors);

    }

    BOWKMeansTrainer bowTrainer(1000);
    cout << "Clustering ... \n";
    Mat vocabulary = bowTrainer.cluster(trainingDescriptors);
    origVocab = vocabulary;

    cout << "Vocabulary Created \n";

}

void VocabBuilder::vocab2File(){

    FileStorage output(vocabOutputFile, FileStorage::WRITE);
    output << "Vocabulary" << origVocab;
    output.release();


}
//Export SVM in this function and open it in testSVM - need to find a way to carry the labels over.

void VocabBuilder::createTrainingData(){

    vector<KeyPoint> keypoints;
    Mat responseHist;
    map<string, Mat> classTrainingData;
    vector<string> classesNames;

    Ptr<FastFeatureDetector> detector(FastFeatureDetector::create());
    Ptr<DescriptorMatcher> matcher(new BFMatcher());
    Ptr<DescriptorExtractor> extractor = SurfDescriptorExtractor::create();
    Ptr<BOWImgDescriptorExtractor> bowImageDescExt(new BOWImgDescriptorExtractor(extractor, matcher));
    bowImageDescExt->setVocabulary(origVocab);

    for (vector<int>::size_type i = 0; i != imageFiles.size(); i++){

        string filename = imageFiles[i];
        string className = VocabBuilder::getClassFromPath(filename);
        cout << className << endl;
        Mat img = imread(filename);
        cout << "Reading image " << i << "\n";
        detector->detect(img, keypoints);
        bowImageDescExt->compute(img, keypoints, responseHist);

        //create classTrainingData

        {
            if (classTrainingData.count(className) == 0){
                classTrainingData[className].create(0, responseHist.cols, responseHist.type());
            }
            classTrainingData[className].push_back(responseHist);
        }

    }

    map<string, Ptr<SVM>>  classifier;
    map<string, string> svmLabelMappings;

    for (map<string, Mat>::iterator iterator = classTrainingData.begin(); iterator != classTrainingData.end(); ++iterator){
        string class_ = (*iterator).first;

        Mat samples(0, responseHist.cols, responseHist.type());
        Mat labels(0,1,CV_32S);
        samples.push_back(classTrainingData[class_]);
        Mat classLabel = Mat::ones(classTrainingData[class_].rows, 1, CV_32S);
        labels.push_back(classLabel);

        //Copy the rest of the labels

        for (map<string, Mat>::iterator iterator1 = classTrainingData.begin(); iterator1 != classTrainingData.end(); ++iterator1) {

            string notClass = (*iterator1).first;
            cout << "Not Class: " << notClass << " Class: " << class_ << endl;
            if (notClass == class_){
                continue;
            }
            samples.push_back(classTrainingData[notClass]);
            classLabel = Mat::zeros(classTrainingData[notClass].rows, 1, CV_32S);
            labels.push_back(classLabel);
        }

        cout << "Training now..." << endl;
        for(int i=0; i<labels.rows; i++){
            for(int j=0; j<labels.cols; j++) {
                printf("labels(%d, %d) = %f \n", i, j, labels.at<float>(i,j));
            }
        }
        //figure out what to do with the string labels mapped to the svm
        Mat samples32f;
        samples.convertTo(samples32f, CV_32F);
        Ptr<TrainData> trainData = TrainData::create(samples, SampleTypes::ROW_SAMPLE, labels);
        classifier[class_] = SVM::create();
        classifier[class_]->train(trainData);
        string classifierFilename = class_ + "_train.xml";
        const char* toCString = classifierFilename.c_str();
        classifier[class_]->save(toCString);
        cout << "Training Done for " << classifierFilename << " - saved xml" << endl;
        svmLabelMappings.insert(pair<string, string>(class_, classifierFilename));

    }

    ofstream of("svmLabels.txt");
    text_oarchive txtOut(of);
    txtOut << svmLabelMappings;

}

//Figure out how to get label strings saved out

void VocabBuilder::testSVM() {

    //Set up BOW Descriptor Extractor
    vector<KeyPoint> keypoints;
    Mat responseHist;
    Ptr<FastFeatureDetector> featureDetector(FastFeatureDetector::create());
    Ptr<DescriptorMatcher> featureMatcher(new BFMatcher());
    Ptr<DescriptorExtractor> extractor = SurfDescriptorExtractor::create();
    Ptr<BOWImgDescriptorExtractor> bowExtractor(new BOWImgDescriptorExtractor(extractor, featureMatcher));
    bowExtractor->setVocabulary(origVocab);

    //Load up SVM and labels into a map
    ifstream inFile(svmLabels);
    text_iarchive inArchive(inFile);
    map<string, string> classLabelMappings;
    inArchive >> classLabelMappings;

    if (initSox() == true) {
      string specPath = boost::filesystem::current_path().string();
      specPath.append("/temp_spectrogram.png");
      // Load Audio File
      sox_format_t *input = openAudio(inputVideoPath.c_str());

      size_t period = 1; // make this a param
      sox_signalinfo_t signalInfo = input->signal;
      sox_rate_t sampleRate = signalInfo.rate;
      sox_uint64_t length = signalInfo.length;
      size_t sampleSize = sampleRate * period;
      size_t samplesTrimmed = 0;

      size_t bufferSize = sampleSize * sizeof(sox_sample_t);
      char buffer[bufferSize];
      int fourcc = CV_FOURCC('H', '2', '6', '4');

      VideoWriter writer;
      Mat frame;
      frame = imread(specPath);
      //writer.open("spectrogram_animation.mp4", fourcc, 1.0, frame.size());
      while (samplesTrimmed < length) {
        cout << length / sampleRate << endl;
        string start = to_string(samplesTrimmed / sampleRate); // seconds
        string end = to_string((samplesTrimmed / sampleRate) + period);
        end.insert(0,"=");
        readAudio(input, buffer, bufferSize, sampleSize);
        const char* png = generateSpectrogram(input, buffer, sampleSize, specPath);
        samplesTrimmed += sampleSize;

        // Test the SVM
        frame = imread(specPath);

        featureDetector->detect(frame, keypoints);
        bowExtractor->compute(frame, keypoints, responseHist);
        cout << "POS :" << samplesTrimmed << endl;
        for (map<string, string>::iterator iterator = classLabelMappings.begin(); iterator != classLabelMappings.end(); ++iterator){
            string className = (*iterator).first;
            const char* svmPath = (*iterator).second.c_str();
            Ptr<SVM> svm = Algorithm::load<SVM>(svmPath);
            int result = svm->predict(responseHist);

            cout << "Class: " << className << " Result : " << result << endl;

        }
        //writer.write(frame);
      }
      sox_close(input);
      //writer.release();
    }
}


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
            cout << "Command line tool that creates BOW vocabularies for a training set of images" << endl;
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
        cout << "Incorrect arguments set. Please run AudioClassifier --help for assistance" << endl;
        return 1;
    }


    return 0;


}
