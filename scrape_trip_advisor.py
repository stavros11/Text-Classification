from utils import trip_advisor_scrapper as tds


#url = "https://www.tripadvisor.com/Hotel_Review-g1188087-d234314-Reviews{}-Kresten_Palace-Koskinou_Kallithea_Rhodes_Dodecanese_South_Aegean.html"
url = "https://www.tripadvisor.com/Hotel_Review-g1012852-d1519291-Reviews{}-The_Kresten_Royal_Villas_Spa-Kallithea_Rhodes_Dodecanese_South_Aegean.html"

scrapper = tds.TripAdvisorReviewScrapper(url)

scrapper.scrape_reviews()
scrapper.save()