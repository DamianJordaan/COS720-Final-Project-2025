import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Tuple

##############################################################################################################
# Simularity Detector
class SimularityDetector:
    def __init__(self, dataframe: pd.DataFrame, text_column: str, id_column: str = "id"):
        """
        Initialize the detector with a DataFrame.
        
        :param dataframe: DataFrame with text documents
        :param text_column: Column containing text
        :param id_column: Column containing document ID (default: 'id')
        """
        if text_column not in dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found.")
        if id_column not in dataframe.columns:
            raise ValueError(f"Column '{id_column}' not found.")

        self.df = dataframe.copy()
        self.text_column = text_column
        self.id_column = id_column
        self.vectorizer = TfidfVectorizer(ngram_range=(3, 5), stop_words='english')  # use 3–5 word phrases
        self.doc_texts = self.df[text_column].astype(str).tolist()
        self.doc_ids = self.df[id_column].tolist()

    def _split_text(self, text: str, window_size: int = 15, step: int = 5) -> List[str]:
        """
        Split the input text into overlapping chunks of `window_size` words.
        """
        words = re.findall(r'\w+', text)
        chunks = [
            " ".join(words[i:i + window_size])
            for i in range(0, len(words) - window_size + 1, step)
        ]
        return chunks if chunks else [" ".join(words)]  # fallback: full text if too short

    def check_simularity(self, input_text: str, threshold: float = 0.6) -> List[Tuple[str, float, any]]:
        """
        Check the input text for plagiarism and return matching subtexts and document IDs.
        
        :param input_text: The text to evaluate
        :param threshold: Similarity threshold
        :return: List of tuples: (matched_subtext, similarity_score, source_doc_id)
        """
        matches = []
        chunks = self._split_text(input_text)
        doc_matrix = self.vectorizer.fit_transform(self.doc_texts)

        for chunk in chunks:
            chunk_vec = self.vectorizer.transform([chunk])
            similarities = cosine_similarity(chunk_vec, doc_matrix).flatten()

            # print("Sims", similarities)
            # print("Chunk", chunk)

            for i, sim in enumerate(similarities):
                if sim >= threshold:
                    matches.append((chunk, sim, self.doc_ids[i]))

        # Remove duplicates (same chunk matched multiple docs with low delta)
        unique_matches = list({(m[0], m[2]): m for m in matches}.values())
        return sorted(unique_matches, key=lambda x: -x[1])  # sorted by similarity


##############################################################################################################
# Check for phishing email similarity
def check_phishing_email_simularity(
    sender: str,
    receiver: str,
    subject: str,
    body: str,
    db_path: str = "phishing_dataset.csv",
    df = None,
):
    """
    Check for phishing email similarity.
    
    :param sender: Sender's email address
    :param receiver: Receiver's email address
    :param subject: Email subject
    :param body: Email body
    :param db_path: Path to the dataset CSV file
    :return: List of tuples with matched subtexts and their similarity scores
    """
    if df is None:
        df = pd.read_csv(db_path)
        # add index as id
        df['id'] = df.index
    df['text'] = df['subject'] + " " + df['body']
    
    detector = SimularityDetector(df, text_column='text', id_column='id')
    
    input_text = f"{subject} {body}"
    return detector.check_simularity(input_text, threshold=0.1)

def format_check_phishing_email_simularity(
    sender: str,
    receiver: str,
    subject: str,
    body: str,
    db_path: str = "phishing_dataset.csv",
    df = None,
    ml_classification_lable: int = 0,
):
    """
    Log the phishing email similarity check.
    
    :param sender: Sender's email address
    :param receiver: Receiver's email address
    :param subject: Email subject
    :param body: Email body
    :param db_path: Path to the dataset CSV file
    """
    if df is None:
        df = pd.read_csv(db_path)
        df['id'] = df.index

    results = check_phishing_email_simularity(sender, receiver, subject, body, db_path, df)
    
    avg_score = np.mean([score for _, score, _ in results]) if results else 0
    results_text = ""
    num_pos_lable_matches = 0
    num_neg_lable_matches = 0
    for subtext, score, doc_id in results:
        doc_label = df[df['id'] == doc_id]['label'].values[0]
        if doc_label == ml_classification_lable:
            results_text += f"\nMatch (doc {doc_id}, similarity {score:.2f}):\n{subtext}"

        if doc_label == 1:
            num_pos_lable_matches += 1
        else:
            num_neg_lable_matches += 1

    lable_decision = 0
    if num_pos_lable_matches > num_neg_lable_matches:
        lable_decision = 1

    return results_text, avg_score, lable_decision
