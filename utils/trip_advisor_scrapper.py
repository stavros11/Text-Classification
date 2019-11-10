import requests
import bs4
import json
import pandas as pd
from typing import Dict, List, Optional


class TripAdvisorReviewScrapper:

  _REVIEW_DATA = ["id", "absoluteUrl", "createdDate", "publishedDate",
                  "locationId", "originalLanguage", "language",
                  "tripInfo/stayDate", "tripInfo/tripType",
                  "helpfulVotes", "title", "text",
                  "rating", "additionalRatings"]

  _REVIEWER_DATA = ["username", "userProfile/userId",
                    "userProfile/hometown/locationId",
                    "userProfile/hometown/location/additionalNames/long"]

  # mgmtResponse/
  _RESPONSE_DATA = ["id", "publishedDate", "language",
                    "username", "connectionToSubject", "text"]

  _REVIEW_TITLES = ([x.split("/")[-1] for x in _REVIEW_DATA] +
                    ["username", "userId", "user_hometownId", "user_hometownName"] +
                    ["response_{}".format(x) for x in _RESPONSE_DATA])

  _SCRIPT_TARGET = "window.__WEB_CONTEXT__"

  def __init__(self, base_url, reviews_per_page=5):
    self.session = requests.Session()
    self.reviews = []

    self.n_reviews = None
    self.reviews_per_page = reviews_per_page
    self.url = base_url

    self.data = self.scrape_data(self.url.format(""))
    print("Found {} reviews for {}.".format(self.n_reviews, self.data["name"]))

  def scrape_reviews(self, start_page: int = 0, last_page: Optional[int] = None,
                     max_reviews: Optional[int] = None):
    counter = start_page * self.reviews_per_page
    max_counter = None if last_page is None else last_page * self.reviews_per_page

    if counter == 0:
      url = self.url.format("")
    else:
      url = "".join([self.url.format("-or{}".format(counter)), "#REVIEWS"])
    revlist = self.get_base(url)["reviewListPage"]["reviews"]

    while revlist:
      for review in revlist:
        self.reviews.append(self.scrape_review(review))

      counter += self.reviews_per_page
      print("{} reviews scrapped.".format(counter))

      if max_reviews is not None and len(self.reviews) >= max_reviews: break
      if max_counter is not None and counter >= max_counter: break

      url = "".join([self.url.format("-or{}".format(counter)), "#REVIEWS"])
      revlist = self.get_base(url)["reviewListPage"]["reviews"]

  def get_base(self, url: str) -> Dict:
    req = self.session.get(url)
    assert req.status_code == 200

    soup = bs4.BeautifulSoup(req.text, "html.parser")
    n = len(self._SCRIPT_TARGET)
    scripts = [script for script in soup.find_all('script')
               if script.text[:n] == self._SCRIPT_TARGET]
    assert len(scripts) == 1

    script_dict = json.loads(scripts[0].text[n + 15:-2])
    base = script_dict["apolloCache"][0]["result"]["locations"][0]

    if "reviewListPage" in base and "totalCount" in base["reviewListPage"]:
      if self.n_reviews is None:
        self.n_reviews = base["reviewListPage"]["totalCount"]
      else:
        assert self.n_reviews == base["reviewListPage"]["totalCount"]

    return base

  def scrape_data(self, url: str) -> Dict:
    base = self.get_base(url)
    data = {"locationId": base["locationId"],
            "name": base["name"],
            "accommodationCategory": base["accommodationCategory"],
            "ratingCounts": base["reviewAggregations"]["ratingCounts"],
            "languageCounts": base["reviewAggregations"]["languageCounts"]}
    return data

  def scrape_review(self, review: Dict) -> List:
    data = [self._nested_get(review, k.split("/")) for k in self._REVIEW_DATA[:-1]]
    # Pay attention to how you log additional ratings
    add_ratings = self._nested_get(review, [self._REVIEW_DATA[-1]])
    # add_ratings is a List of Dicts
    data.append({r["ratingLabel"]: r["rating"] for r in add_ratings})

    for k in self._REVIEWER_DATA:
      data.append(self._nested_get(review, k.split("/")))

    for k in self._RESPONSE_DATA:
      actual_k = "mgmtResponse/{}".format(k)
      data.append(self._nested_get(review, actual_k.split("/")))

    return data

  def _nested_get(self, data: Dict, keys: List[str]):
    if not (isinstance(data, dict) and keys[0] in data):
      return None

    if len(keys) > 1:
      return self._nested_get(data[keys[0]], keys[1:])
    return data[keys[0]]

  @property
  def lower_name(self) -> str:
    return "_".join(self.data["name"].lower().split(" "))

  def to_dataframe(self) -> pd.DataFrame:
    return pd.DataFrame(self.reviews, columns=self._REVIEW_TITLES)

  def save(self):
    name = "{}_{}reviews".format(self.lower_name, len(self.reviews))
    with open("{}.txt".format(name), "w") as file:
      json.dump(self.data, file)

    df = self.to_dataframe()
    df.to_csv("{}.csv".format(name), index=False)