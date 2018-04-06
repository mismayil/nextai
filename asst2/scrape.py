import bs4
from bs4 import BeautifulSoup
from urllib import request

CAVISA_URL = 'https://www.canadavisa.com/'
CAVISA_FAQ_URL = 'https://www.canadavisa.com/canada-immigration-questions-faq.html'

def get_soup(url):
    response = request.urlopen(url)
    page = response.read()
    soup = BeautifulSoup(page, 'html5lib')
    return soup

soup = get_soup(CAVISA_FAQ_URL)
divs = soup.find_all(class_='row faq-box')

links = []
for div in divs:
    links.extend(div.find_all('a'))
    
with open('faq.txt', 'a') as file:
    for link in links:
        print(link)
        if link.get('href'): soup = get_soup(link['href'])
        article = soup.find(attrs={'itemprop':'articleBody'})

        text = ''
        for child in article.children:
            if isinstance(child, bs4.element.NavigableString):
                text += child
            else:
                text += child.get_text()

        file.write(text)
