getwd()
library(data.table)

path = "GitHub/phish_setlistbot/datacollection/setlists/2"
yearfolders = list.files(path)

df = data.table()
for (item in yearfolders){
  yearpath = paste(path, item, sep="/")
  shows = list.files(yearpath)
  for (show in shows){
    data = fread(paste(yearpath, show, sep="/"))
    data$Date = substr(show, 1, nchar(show)-4)
    df = rbind(df, data)
  }
}

getwd()
write.csv(df, "GitHub/phish_setlistbot/datacollection/setlists/combined/2.csv")

