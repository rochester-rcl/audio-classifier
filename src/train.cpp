#include "audio-classifier.cpp"

int main(int argc, const char* argv[]) {
  //Get command line options
  options_description description("Options");
  description.add_options()
  ("help,h", "Print help messages")
  ("train_dir,t", value<string>(), "Directory to recursively search for folders with training images")
  ("vocab_file,v", value<string>(), "Path to the SVM vocabulary that will be saved (YAML)");


  variables_map variables;
  store(parse_command_line(argc, argv, description), variables);
  try {
      if(variables.count("help")){
          cout << "Command line tool to train an SVM classifier" << endl;
          cout << "Usage: train --train_dir my-path/my-dir/ --vocab path/to/vocab.yml" << endl << endl;
          return 0;
      }
      notify(variables);
  } catch (boost::program_options::error& e) {
      cerr << "ERROR: " << e.what() << endl;
      cerr << description << endl;
      return 1;
  }

  if (variables.count("train_dir") && variables.count("vocab_file")) {
      string directoryPath = variables["train_dir"].as<string>();
      string vocabOutputFile = variables["vocab_file"].as<string>();
      VocabBuilder vocab(directoryPath, vocabOutputFile); // get files in provided directory, output YAML file
      vocab.createVocab();
      vocab.createTrainingData();
      vocab.vocab2File();
  } else {
      cout << "Incorrect arguments set. Please run audio-classifier --help for assistance" << endl;
      return 1;
  }
}
