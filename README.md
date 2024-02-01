# Turkish-Movie-Classification-with-Neural-Networks

## About the project:
The goal of the project is showing my Deep Learning skills and brighten my skills in Natural Language Processing. Turkish movies and serials are well-known worldwide, so I wanted to work on Turkish movies, also there are not many projects on Turkish language out here.

I collected, cleaned and processed the dataset in order to train big variety of models from most popular Machine Learning - Deep Learning libraries such as Tensorflow, Keras, PyTorch and Scikit-Learn. Also finetuned a pretrained model from HuggingFace. If you want to reach out to me, here is my [LinkedIn](www.linkedin.com/in/chanyalcin).

## About the dataset:
I self collected everything in this dataset, first I choosed the movie types I want to work on, then found 
movies on Youtube and collected them into 5 different playlists (because the library I used can only process 50 movies at once)

## Collecting the Data:
I used googleapiclient library to access youtube v3 api
I got an api key from youtube

I extracted every video's url and title by using api

I used pytube to download videos as .mp3 files,
* I modified 'yt\Lib\site-packages\pytube\streams.py' file's 311th line, I added 'if not os.path.exists(file_path):' to handle the case where we do not have file_path exists*
* I modified 'yt\Lib\site-packages\pytube\innertube.py' file's 223th line from 'def __init__(self, client='ANDROID_MUSIC', use_oauth=False, allow_cache=True): 
' to '    def __init__(self, client='ANDROID', use_oauth=False, allow_cache=True):
'* # https://stackoverflow.com/a/76780768/21653250 #

I used OpenAı's whisper to transribe videos one by one and saved into a pandas dataframe, "name","transcript" columns.

## How did I handle such a big text dataset?

I defined custom functions in function.py
remove_punctuation(), contains_non_turkish(), remove_words_without_non_turkish(), remove_substring(), remove_consecutive_duplicates()

I cleaned the dataset, dropped dirty rows, applied predefined functions on values.
checked if we have any Na values,
plotted the pie chart of labels















