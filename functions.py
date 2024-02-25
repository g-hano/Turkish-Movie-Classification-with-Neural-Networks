import pandas as pd
import string 
import re
import unicodedata

def remove_punctuation(text: str) -> str:
    """
    Simply removes any punctuation character from the gives text.
    """
    translator = str.maketrans('', '', string.punctuation)
    text = re.sub(r'[^\w\s]', '', text)
    return text.translate(translator)
    
def contains_non_turkish(word: str) -> bool:
    """
    Checks if the character in the word is valid 'LATIN' or not.
    """
    for char in word:
        if 'LATIN' not in unicodedata.name(char, ''):
            return False
    return True
    
def remove_non_turkish(text: str) -> str:
    """
    Removes the word if it contains any non-Turkish character.
    """
    words = text.split()
    words = [word for word in words if contains_non_turkish(word)]
    return ' '.join(words)
    
def remove_substring(df: pd.DataFrame) -> pd.DataFrame:
    """
    OpenAI' Whisper is not perfect, some sentences appear in the transcript due to Whisper's fault or some films has intros/ads in it.
    """
    unwanted_text = ["intro ", "müzik ", 
             "bu dizinin betimlemesi trt tarafından sesli betimleme derneğine yaptırılmıştır ",
             "altyazı mk ", "ileri ", "gel ", 
             "yerli film tutkunları sizler için fanatik film yerli kanalı yayında yeni eski harika türk filmleri sizleri bekliyor kanalımıza abone olmayı unutmayın ",
            "evet ",
             "izlediğiniz için teşekkür ederim ",
                     "sesli betimleme metin yazarı ve seslendiren ",
                     "altyazı ",
                     "altyazı mk",
                     
            ]
    for unwanted in unwanted_text:
        df["transcript"] = df["transcript"].str.replace(f"{unwanted}", "", case=False)

    return df
    
def remove_consecutive_duplicates(text:str ) -> str:
    """
    Removes the duplicates if it appears more than once.
    """
    words = text.split()
    cleaned_words = [words[i] for i in range(len(words)) if i == 0 or (words[i] != words[i-3] and words[i] != words[i-2] and words[i] != words[i-1])]
    return ' '.join(cleaned_words)
