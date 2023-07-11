rm(list=ls())

library(tidyverse)
library(rvest)

url = "https://www.detik.com/"

url_all = url %>% read_html() %>% html_nodes("a") %>% html_attr("href")
url_all = url_all[grepl("news.detik.com/berita|oto.detik.com/otosport/",url_all)] %>% unique()

baca_donk = function(i){
  baca   = url_all[i] %>% read_html()
  title  = baca %>% html_nodes(".detail__title") %>% html_text()
  text   = baca %>% html_nodes(".itp_bodycontent") %>% html_text()
  output = data.frame(title,text)
}

hasil = vector("list",length(url_all))

for(i in 1:length(url_all)){
  hasil[[i]] = baca_donk(i)
  print(i)
}

data = do.call(rbind,hasil)
write.csv(data,"data_4.csv")