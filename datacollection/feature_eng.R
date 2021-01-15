library(data.table)
library(progress)
data1= fread("GitHub/phish_setlistbot/datacollection/setlists/combined/1.csv")
data2= fread("GitHub/phish_setlistbot/datacollection/setlists/combined/2.csv")
data3= fread("GitHub/phish_setlistbot/datacollection/setlists/combined/3.csv")

data = rbind(data1, data2, data3)
write.csv(data, "GitHub/phish_setlistbot/datacollection/setlists/combined/all.csv", row.names=FALSE)

########
#make date lower case
names(data)[3] <- "date"

#######
# number each show to have reference
#######
data$show = 1
show_n = 1
currentdate = data$date[1]

for (ii in 1:dim(data)[1]) {
  data$show[ii] = show_n 
  
  if (data$date[ii] != currentdate){
    show_n = show_n + 1
    currentdate = data$date[ii]
  }
}

###
# calculate gap for each song
###

data <- as.data.frame(data) #this loop wasn't working in data.table fomat
data$gap = 0 #initialize gap

for (song in unique(data$Song)){
  temp = data[data$Song == song,]
  temp
  if (dim(temp)[1] != 1){
    for (ii in 2:dim(temp)[1]){
      temp$gap[ii] = temp$show[ii] - temp$show[ii-1]
    }  
  }
  
  data[data$Song == song,]$gap = temp$gap
  
}

#check out gaps
hist(data[data$gap<=25,]$gap, breaks = c(0,5,10,15,20,25))

head(data)

data$date[1]
data$date[1] + 30


###
# calculate real number of days since last played
###
data$realGap = 0 #initialize gap

temp$date[2] - temp$date[1]
temp[1:2,]

for (song in unique(data$Song)){
  temp = data[data$Song == song,]
  temp
  if (dim(temp)[1] != 1){
    for (ii in 2:dim(temp)[1]){
      temp$realGap[ii] = temp$date[ii] - temp$date[ii-1]
    }  
  }
  data[data$Song == song,]$realGap = temp$realGap
}

###
# create "tour" column that will iterate every 10 day gap between occurring dates
###
pb <- progress_bar$new(total = dim(data)[1])


data$tour = 1
head(data)
current_tour = 1

for (ii in 2:dim(data)[1]) {
  
  if (data$date[ii] - data$date[ii-1] > 10){
    current_tour = current_tour + 1
  }
  data$tour[ii] = current_tour
  pb$tick()
}

###
# 
###




###
# write each time we add a feature to save our progress
###
write.csv(data, "GitHub/phish_setlistbot/datacollection/final.csv", row.names=FALSE)

