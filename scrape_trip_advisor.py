import argparse
from utils import trip_advisor_scrapper as tds


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--url', type=str)
  parser.add_argument('--max-reviews', type=int, default=None)

  args = parser.parse_args()
  url = args.url

  # Add {} after Reviews
  n = url.find("Reviews") + len("Reviews")
  url = "".join([url[:n], "{}", url[n:]])
  print("Given url:")
  print(url)
  print()

  scrapper = tds.TripAdvisorReviewScrapper(url)
  scrapper.scrape_reviews(max_reviews=args.max_reviews)
  scrapper.save()