library(dplyr)
imdb_file = "../data/imdb_top_1000.csv"
imdb_table = data.table::fread(imdb_file) %>% 
  tibble::tibble() %>% 
  dplyr::transmute(name = Series_Title,
                   year = Released_Year,
                   gross = as.numeric(gsub(",", "",Gross)),
                   runtime = as.numeric(stringi::stri_extract(Runtime,regex = "\\d+")),
                   genre =  sub(",.*", "", Genre),
                   imdb = as.numeric(IMDB_Rating),
                   metascore = as.numeric(Meta_score),
                   director = Director,
                   votes = as.numeric(No_of_Votes),
                   title_length = name %>% nchar()) %>% 
  na.omit() %>% 
  dplyr::mutate(genre_num = dplyr::case_when(
    genre =="Drama" ~1,
    genre =="Crime" ~2,
    genre =="Action"~3,
    genre =="Biography"~4,
    genre =="Western"~5,
    genre =="Comedy" ~ 6,
    genre =="Adventure"~ 7,
    genre =="Animation"~ 8,
    genre =="Horror"~9,
    genre =="Mystery"~10,
    genre =="Film-Noir"~11,
    genre =="Fantasy" ~12,
    genre =="Family" ~13,
    genre =="Thriller"~14,
    TRUE~15
  ))

#particular cases
imdb_table$year[imdb_table$year=="PG"] <- 1995

data.table::fwrite(imdb_table, file= "imdb_processed.tsv",sep = "\t", col.names = T)
