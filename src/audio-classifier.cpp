#include "audio-classifier.h"

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

int VocabBuilder::trainingSetFromAudio(string audioFile, string outFolder, float sampleLength) {
  // Load Audio File
  if (initSox() == true) {

    sox_format_t *input = openAudio(audioFile.c_str());

    size_t period = 1;
    sox_signalinfo_t signalInfo = input->signal;
    sox_rate_t sampleRate = signalInfo.rate;
    sox_uint64_t length = signalInfo.length;
    cout << sampleLength << endl;
    size_t sampleSize = (sampleLength > 1.0) ? (sampleRate * period) * sampleLength : (sampleRate * period) / (1.0f / sampleLength);

    size_t samplesTrimmed = 0;

    size_t bufferSize = sampleSize * sizeof(sox_sample_t);
    char buffer[bufferSize];
    try {
      while(samplesTrimmed < length) {
        string specUUID = to_string(random_generator()());
        string specPath = outFolder + "/" + specUUID + ".png";
        string start = to_string(samplesTrimmed / sampleRate); // seconds
        string end = to_string((samplesTrimmed / sampleRate) + period);
        end.insert(0,"=");
        readAudio(input, buffer, bufferSize, sampleSize);
        const char* png = generateSpectrogram(input, buffer, sampleSize, specPath);
        samplesTrimmed += sampleSize;
      }
    } catch(exception& e) {
      cout << e.what() << endl;
      return 1;
    }
    sox_close(input);
  } else {
    return 1;
  }
  return 0;
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

void VocabBuilder::loadVocabFile(FileStorage vocabFile) {
  vocabFile["Vocabulary"] >> origVocab;
}

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
    // borrowed from https://github.com/goruck/bow/blob/master/src/bow-classification.cpp
    int clusterSize = 1000;
    int attempts = 3;
    int flags = cv::KMEANS_RANDOM_CENTERS;
    TermCriteria terminate_criterion(TermCriteria::EPS | TermCriteria::COUNT, 10, 1.0);

    BOWKMeansTrainer bowTrainer = BOWKMeansTrainer(clusterSize, terminate_criterion, attempts, flags);
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

    string absPath = absolute(path(trainFilesDirectory)).string();
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

        string classifierFilename = absPath + "/" + class_ + "_train.xml";
        const char* toCString = classifierFilename.c_str();
        cout << classifierFilename << endl;
        classifier[class_]->save(toCString);
        cout << "Training Done for " << classifierFilename << " - saved xml" << endl;
        svmLabelMappings.insert(pair<string, string>(class_, classifierFilename));

    }
    // add path to vocab too so we can have 2 params in main instead of 3
    ofstream of(absPath + "/svmLabels.txt");
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
      size_t sampleSize = (sampleRate * period);

      size_t samplesTrimmed = 0;

      size_t bufferSize = sampleSize * sizeof(sox_sample_t);
      char buffer[bufferSize];
      int fourcc = CV_FOURCC('H', '2', '6', '4');

      VideoWriter writer;
      Mat frame;
      frame = imread(specPath);
      //writer.open("spectrogram_animation.mp4", fourcc, 1.0, frame.size());
      while (samplesTrimmed < length) {
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
            Mat results;
            int result = svm->predict(responseHist, results, true);

            cout << "Class: " << className << " Result : " << result << ' ' << results.at<float>(0,0) << endl;

        }
        //writer.write(frame);
      }
      sox_close(input);
      //writer.release();
    }
}
