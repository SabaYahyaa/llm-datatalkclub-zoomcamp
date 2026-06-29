"""
Ingest data form a source
    
"""
import requests
from minsearch import Index


def load_faq_data():
    docs_url = "https://datatalks.club/faq/json/courses.json"
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses_raw:
        course_url = f"""{url_prefix}{course["path"]}"""

        course_response = requests.get(course_url)
        course_response.raise_for_status()
        # get a list of dictionaries from a specific course
        course_data = course_response.json()
        # extend the list
        documents.extend(course_data)
    return documents



def build_index(documents):
    # 1.2. Get your data set, documents (a list of dictionaries)
    index = Index(
        text_fields=["question", "section", "answer"], # fields that conatin useful information to get the answer (fields that are used to perfom search)
        keyword_fields=["course"], # this is used when I want to filter, for example select * from documents where section="Machine Learning for Classification" then I will ignore all the courses and I will use only that section,
    )
    # fit with the documents (json file that contains all the prepared data)
    index.fit(documents)
    return index


if __name__ == "__main__":
    print('y')
