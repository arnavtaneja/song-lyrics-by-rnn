#search for song
import requests
from bs4 import BeautifulSoup
import re
from lxml import html
from urllib.request import urlopen



ENTERPERSONNAMEHERE = 'drake'
artist = ENTERPERSONNAMEHERE.replace(' ','').lower()


artistURL = 'https://www.azlyrics.com/'+ artist[0]+ '/' + artist + '.html'
page = requests.get(artistURL)
temp = BeautifulSoup(page.text, "html.parser") # Extract the page's HTML as a string



html=temp.text

##print(html)

songnames=[]

indexofsong =0
i = 1
base = 100
while i==1:
    indexofsong = html.find("{s:",base)
   ## print(indexofsong)
    tempstring =""
    if(indexofsong == -1):
        i=-1
        break

    for x in range(4, 100):
        if(html[indexofsong+x] != '"'):
            tempstring += html[indexofsong+x]
        else:
            x = 500
            songnames.append(tempstring)
            base = indexofsong+len(tempstring)+5
            break

    ## print(tempstring)

for items in songnames:
    print(items)

##print(len(songnames))

