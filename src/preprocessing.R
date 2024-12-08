library(dplyr)
imdb_file = "imdb_top_1000.csv"
imdb_table = data.table::fread(imdb_file) %>% 
  tibble::tibble() %>% 
  dplyr::transmute(name = Series_Title,
                   year = Released_Year,
                   budget = as.numeric(gsub(",", "",Gross)),
                   runtime = stringi::stri_extract(Runtime,regex = "\\d+"),
                   genre =  Genre,
                   imdb = as.numeric(IMDB_Rating),
                   #metascore = as.numeric(Meta_score),
                   director = Director,
                   votes = No_of_Votes) %>% 
  na.omit()

data.table::fwrite(imdb_table, file= "imdb_processed.tsv",sep = "\t", col.names = T)
