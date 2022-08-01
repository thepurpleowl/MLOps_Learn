import os
from fastapi.testclient import TestClient
from .app.server import app

import pytest

os.chdir('app')
client = TestClient(app)

""" 
This part is optional. 

We've built our web application, and containerized it with Docker.
But imagine a team of ML engineers and scientists that needs to maintain, improve and scale this service over time. 
It would be nice to write some tests to ensure we don't regress! 

  1. `Pytest` is a popular testing framework for Python. If you haven't used it before, take a look at https://docs.pytest.org/en/7.1.x/getting-started.html to get started and familiarize yourself with this library.
   
  2. How do we test FastAPI applications with Pytest? Glad you asked, here's two resources to help you get started:
    (i) Introduction to testing FastAPI: https://fastapi.tiangolo.com/tutorial/testing/
    (ii) Testing FastAPI with startup and shutdown events: https://fastapi.tiangolo.com/advanced/testing-events/

"""

def test_root():
    """
    [TO BE IMPLEMENTED]
    Test the root ("/") endpoint, which just returns a {"Hello": "World"} json response
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_predict_empty():
    """
    [TO BE IMPLEMENTED]
    Test the "/predict" endpoint, with an empty request body
    """
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_en_lang():
    """
    [TO BE IMPLEMENTED]
    Test the "/predict" endpoint, with an input text in English (you can use one of the test cases provided in README.md)
    """
    # Here event handlers (startup and shutdown) to run in your tests
    # hence use TestClient with a `with` statement
    # https://fastapi.tiangolo.com/advanced/testing-events/
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "source": "Boston Globe",
                "url": "http://www.boston.com/business/articles/2005/01/04/mike_wallace_subpoenaed?rss_id=BostonGlobe--BusinessNews",
                "title": "Mike Wallace subpoenaed",
                "description": "Richard Scrushy once sat down to talk with 60 Minutes correspondent Mike Wallace about allegations that Scrushy started a huge fraud while chief executive of rehabilitation giant HealthSouth Corp. Now, Scrushy wants Wallace to do the talking."
                },
        )
        assert response.status_code == 200
        assert response.json() == {
            "scores": pytest.approx(
                {
                    "Business": 0.23508955472074963,
                    "Entertainment": 0.5724817112974444,
                    "Health": 0.0823957248580479,
                    "Music Feeds": 0.003701017629065236,
                    "Sci/Tech": 0.03923397760096245,
                    "Software and Developement": 0.009537954019880498,
                    "Sports": 0.051965575250882294,
                    "Toons": 0.00559448462296766
                }
            ),
            "label": "Entertainment",
        }


def test_predict_es_lang():
    """
    [TO BE IMPLEMENTED]
    Test the "/predict" endpoint, with an input text in Spanish. 
    Does the tokenizer and classifier handle this case correctly? Does it return an error?
    """
    pass


def test_predict_non_ascii():
    """
    [TO BE IMPLEMENTED]
    Test the "/predict" endpoint, with an input text that has non-ASCII characters. 
    Does the tokenizer and classifier handle this case correctly? Does it return an error?
    """
    pass