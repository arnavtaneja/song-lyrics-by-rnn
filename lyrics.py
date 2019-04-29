from songs import songnames
from songs import artist

from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests

def printlyrics(musicperson, name):



    songname = name.replace(' ','').lower()
    songname = songname.replace("'",'')
    songname = songname.replace("&quot;",'')
    songname = songname.replace("(",'')
    songname = songname.replace(")",'')
    songname = songname.replace(".",'')
    songname = songname.replace(' ','').lower()

    ##print(songname)


    songURL = 'https://www.azlyrics.com/lyrics/' + musicperson + '/' + songname + '.html'
    page = requests.get(songURL)

    soup = BeautifulSoup(page.text, 'html.parser') # Extract the page's HTML as a string


    html=soup.text
    ## print(html)




    ##print(html[10])
    end = html.find("-",10)
    name = html[10:end-1]



    start=html.find(name + ' Lyrics')


    ##print(start)

    endpt = html.find("if  ( /Android")

    ## print(endpt)


    print(html[start:endpt-5])


for x in range(0,len(songnames)):
    printlyrics(artist, songnames[x])

