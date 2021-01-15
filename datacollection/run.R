library(phishr)

phishnet_key <- "B93C5925297AB520755D"


start <- as.Date(0, origin="1983-12-02")
end   <- as.Date(0, origin="2020-04-01")


theDate <- start

showdates <- list()
showlengths <- c()


bigiter = 1
while(theDate <= end) 
{
  if (bigiter %% 30 == 1)
  {
    print(theDate)
  }
  setlist <- phishr::pn_get_setlist(phishnet_key, theDate)
  if (!is.na(setlist))
  {
    temp <- theDate
    showdates <- append(showdates, theDate)
    showlengths <- append(showlengths, dim(setlist)[1])
  }
  theDate <- theDate + 1
  bigiter <- bigiter + 1
}


phishr::pn_get_setlist(phishnet_key, "2019-02-20")

showdates
showlengths

plot(showdates[showlengths > 5], showlengths[showlengths > 5])
plot(showdates, showlengths)
plot(showdates[showdates > "2009-01-01"], showlengths[showdates > "2009-01-01"])

showdates[showlengths <= 5]

temp <- phishr::pn_get_setlist(phishnet_key, "2018-12-30")


data = data.frame(showdates, showlengths)
write.csv(data, "c:/Users/rober/Documents/phish_setlistbot/datacollection/showlengths.csv", quote=FALSE, row.names=FALSE)





"""
Script to grab all setlists and save a csv for each show, subdivided by era and year
"""

library(lubridate)
phishnet_key <- "B93C5925297AB520755D"

start <- as.Date(0, origin="1983-01-01")
end   <- as.Date(0, origin="2001-01-01")


theDate <- start #current date for loops
notifier = theDate + months(1)
theYear <- year(theDate)

while(theDate <= end) 
{
  if (theDate > notifier)
  {
    print(theDate)
    notifier = notifier + months(1)
  }
  if (year(theDate) == theYear + 1)
  {
    theYear = theYear + 1
  }
  
  setlist <- phishr::pn_get_setlist(phishnet_key, theDate)
  
  if (!is.na(setlist)) #IF THERE WAS ACTUALLY A SHOW
  {
    setlist <- setlist[,1:2]
    write.csv(setlist, paste(paste("c:/Users/rober/Documents/GitHub/phish_setlistbot/datacollection/setlists/1", theYear, theDate, sep="/"), "csv", sep="."), row.names=FALSE)
  }
  theDate <- theDate + 1
}

theDate




