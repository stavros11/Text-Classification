from utils import trip_advisor_scrapper as tds


#url = "https://www.tripadvisor.com/Hotel_Review-g1188087-d234314-Reviews{}-Kresten_Palace-Koskinou_Kallithea_Rhodes_Dodecanese_South_Aegean.html"
#url = "https://www.tripadvisor.com/Hotel_Review-g1012852-d1519291-Reviews{}-The_Kresten_Royal_Villas_Spa-Kallithea_Rhodes_Dodecanese_South_Aegean.html"
#url = "https://www.tripadvisor.com/Hotel_Review-g1188087-d2452962-Reviews{}-Stavros_Melathron_Studios-Koskinou_Kallithea_Rhodes_Dodecanese_South_Aegean.html"
#url = "https://www.tripadvisor.com/Hotel_Review-g776010-d572243-Reviews{}-Lindian_Village-Lardos_Rhodes_Dodecanese_South_Aegean.html"
url = "https://www.tripadvisor.com/Hotel_Review-g635612-d296796-Reviews{}-Blue_Sea_Beach_Resort-Faliraki_Rhodes_Dodecanese_South_Aegean.html"

scrapper = tds.TripAdvisorReviewScrapper(url)

scrapper.scrape_reviews()
scrapper.save()