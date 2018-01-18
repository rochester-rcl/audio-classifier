#include "audio-classifier.cpp"


int main(int argc, const char* argv[]) {

    //Get command line options
    options_description description("Options");
    description.add_options()
    ("help,h", "Print help messages")
    ("dir,d", value<string>(), "Directory to save spectrograms to")
    ("input,i", value<string>(), "Input audio file")
    ("sample_length,s", value<float>(), "sample length in seconds [float]");


    variables_map variables;
    store(parse_command_line(argc, argv, description), variables);
    try {
        if(variables.count("help")){
            cout << "Command line tool that creates spectrograms from n chunks of an audio file" << endl;
            cout << "Usage: create-training-set --input audio.wav --dir my-path/my-dir/ --sample_length 2" << endl << endl;
            return 0;
        }
        notify(variables);
    } catch (boost::program_options::error& e) {
        cerr << "ERROR: " << e.what() << endl;
        cerr << description << endl;
        return 1;
    }

    if (variables.count("dir") && variables.count("input") && variables.count("sample_length")) {
      string input = variables["input"].as<string>();
      string dir = variables["dir"].as<string>();
      float sampleLength = variables["sample_length"].as<float>();
      return VocabBuilder::trainingSetFromAudio(input, dir, sampleLength);
    }
}
