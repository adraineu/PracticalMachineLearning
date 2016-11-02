# Practical Machine Learning Course Project
Adraine Upshaw  
October 29, 2016  
Synopsis: 
-------------------------------------
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, I will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. Participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways(classe variable). The project will outline the steps used to develop a prediction model for the 5 different ways. 
More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

Data Processing (loading and preprocessing the data):
-----------------------------------------------------

```r
options(scipen = 2, digits = 0)
setwd("~/DataScience/Pratical Machine Learning")
#setwd("H:/DataScientist/practical-machine-learning")
library(caret); library(rattle); library(rpart); library(rpart.plot)
library(randomForest);library(gbm);library(ggplot2)
library(plyr);library(knitr) 
if(!file.exists("./data")){dir.create("./data")}
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
if (!file.exists("./data/pml-training.csv")) {download.file(fileUrl, destfile ="./data/pml-training.csv")
}
training <- read.csv("./data/pml-training.csv", header = TRUE, 
                         na.strings = c("","NA","#DIV/0!"))

fileUrl2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if (!file.exists("./data/pml-testing.csv")) {download.file(fileUrl2, destfile ="./data/pml-testing.csv")
}
testing <- read.csv("./data/pml-testing.csv", header = TRUE, 
                    na.strings = c("","NA","#DIV/0!"))
str(training)
```

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_belt.1    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ kurtosis_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ kurtosis_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
```

```r
summary(training)
```

```
##        X            user_name    raw_timestamp_part_1 raw_timestamp_part_2
##  Min.   :    1   adelmo  :3892   Min.   :1322489605   Min.   :   294      
##  1st Qu.: 4906   carlitos:3112   1st Qu.:1322673099   1st Qu.:252912      
##  Median : 9812   charles :3536   Median :1322832920   Median :496380      
##  Mean   : 9812   eurico  :3070   Mean   :1322827119   Mean   :500656      
##  3rd Qu.:14717   jeremy  :3402   3rd Qu.:1323084264   3rd Qu.:751891      
##  Max.   :19622   pedro   :2610   Max.   :1323095081   Max.   :998801      
##                                                                           
##           cvtd_timestamp  new_window    num_window    roll_belt  
##  28/11/2011 14:14: 1498   no :19216   Min.   :  1   Min.   :-29  
##  05/12/2011 11:24: 1497   yes:  406   1st Qu.:222   1st Qu.:  1  
##  30/11/2011 17:11: 1440               Median :424   Median :113  
##  05/12/2011 11:25: 1425               Mean   :431   Mean   : 64  
##  02/12/2011 14:57: 1380               3rd Qu.:644   3rd Qu.:123  
##  02/12/2011 13:34: 1375               Max.   :864   Max.   :162  
##  (Other)         :11007                                          
##    pitch_belt     yaw_belt    total_accel_belt kurtosis_roll_belt
##  Min.   :-56   Min.   :-180   Min.   : 0       Min.   :-2        
##  1st Qu.:  2   1st Qu.: -88   1st Qu.: 3       1st Qu.:-1        
##  Median :  5   Median : -13   Median :17       Median :-1        
##  Mean   :  0   Mean   : -11   Mean   :11       Mean   : 0        
##  3rd Qu.: 15   3rd Qu.:  13   3rd Qu.:18       3rd Qu.: 0        
##  Max.   : 60   Max.   : 179   Max.   :29       Max.   :33        
##                                                NA's   :19226     
##  kurtosis_picth_belt kurtosis_yaw_belt skewness_roll_belt
##  Min.   :-2          Mode:logical      Min.   :-6        
##  1st Qu.:-1          NA's:19622        1st Qu.: 0        
##  Median : 0                            Median : 0        
##  Mean   : 4                            Mean   : 0        
##  3rd Qu.: 3                            3rd Qu.: 0        
##  Max.   :58                            Max.   : 4        
##  NA's   :19248                         NA's   :19225     
##  skewness_roll_belt.1 skewness_yaw_belt max_roll_belt   max_picth_belt 
##  Min.   :-8           Mode:logical      Min.   :-94     Min.   : 3     
##  1st Qu.:-1           NA's:19622        1st Qu.:-88     1st Qu.: 5     
##  Median : 0                             Median : -5     Median :18     
##  Mean   : 0                             Mean   : -7     Mean   :13     
##  3rd Qu.: 1                             3rd Qu.: 18     3rd Qu.:19     
##  Max.   : 7                             Max.   :180     Max.   :30     
##  NA's   :19248                          NA's   :19216   NA's   :19216  
##   max_yaw_belt   min_roll_belt   min_pitch_belt   min_yaw_belt  
##  Min.   :-2      Min.   :-180    Min.   : 0      Min.   :-2     
##  1st Qu.:-1      1st Qu.: -88    1st Qu.: 3      1st Qu.:-1     
##  Median :-1      Median :  -8    Median :16      Median :-1     
##  Mean   : 0      Mean   : -10    Mean   :11      Mean   : 0     
##  3rd Qu.: 0      3rd Qu.:   9    3rd Qu.:17      3rd Qu.: 0     
##  Max.   :33      Max.   : 173    Max.   :23      Max.   :33     
##  NA's   :19226   NA's   :19216   NA's   :19216   NA's   :19226  
##  amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
##  Min.   :  0         Min.   : 0           Min.   :0         
##  1st Qu.:  0         1st Qu.: 1           1st Qu.:0         
##  Median :  1         Median : 1           Median :0         
##  Mean   :  4         Mean   : 2           Mean   :0         
##  3rd Qu.:  2         3rd Qu.: 2           3rd Qu.:0         
##  Max.   :360         Max.   :12           Max.   :0         
##  NA's   :19216       NA's   :19216        NA's   :19226     
##  var_total_accel_belt avg_roll_belt   stddev_roll_belt var_roll_belt  
##  Min.   : 0           Min.   :-27     Min.   : 0       Min.   :  0    
##  1st Qu.: 0           1st Qu.:  1     1st Qu.: 0       1st Qu.:  0    
##  Median : 0           Median :116     Median : 0       Median :  0    
##  Mean   : 1           Mean   : 68     Mean   : 1       Mean   :  8    
##  3rd Qu.: 0           3rd Qu.:123     3rd Qu.: 1       3rd Qu.:  0    
##  Max.   :16           Max.   :157     Max.   :14       Max.   :201    
##  NA's   :19216        NA's   :19216   NA's   :19216    NA's   :19216  
##  avg_pitch_belt  stddev_pitch_belt var_pitch_belt   avg_yaw_belt  
##  Min.   :-51     Min.   :0         Min.   : 0      Min.   :-138   
##  1st Qu.:  2     1st Qu.:0         1st Qu.: 0      1st Qu.: -88   
##  Median :  5     Median :0         Median : 0      Median :  -7   
##  Mean   :  1     Mean   :1         Mean   : 1      Mean   :  -9   
##  3rd Qu.: 16     3rd Qu.:1         3rd Qu.: 0      3rd Qu.:  14   
##  Max.   : 60     Max.   :4         Max.   :16      Max.   : 174   
##  NA's   :19216   NA's   :19216     NA's   :19216   NA's   :19216  
##  stddev_yaw_belt  var_yaw_belt    gyros_belt_x  gyros_belt_y  gyros_belt_z
##  Min.   :  0     Min.   :    0   Min.   :-1    Min.   :-1    Min.   :-1   
##  1st Qu.:  0     1st Qu.:    0   1st Qu.: 0    1st Qu.: 0    1st Qu.: 0   
##  Median :  0     Median :    0   Median : 0    Median : 0    Median : 0   
##  Mean   :  1     Mean   :  107   Mean   : 0    Mean   : 0    Mean   : 0   
##  3rd Qu.:  1     3rd Qu.:    0   3rd Qu.: 0    3rd Qu.: 0    3rd Qu.: 0   
##  Max.   :177     Max.   :31183   Max.   : 2    Max.   : 1    Max.   : 2   
##  NA's   :19216   NA's   :19216                                            
##   accel_belt_x   accel_belt_y  accel_belt_z  magnet_belt_x magnet_belt_y
##  Min.   :-120   Min.   :-69   Min.   :-275   Min.   :-52   Min.   :354  
##  1st Qu.: -21   1st Qu.:  3   1st Qu.:-162   1st Qu.:  9   1st Qu.:581  
##  Median : -15   Median : 35   Median :-152   Median : 35   Median :601  
##  Mean   :  -6   Mean   : 30   Mean   : -73   Mean   : 56   Mean   :594  
##  3rd Qu.:  -5   3rd Qu.: 61   3rd Qu.:  27   3rd Qu.: 59   3rd Qu.:610  
##  Max.   :  85   Max.   :164   Max.   : 105   Max.   :485   Max.   :673  
##                                                                         
##  magnet_belt_z     roll_arm      pitch_arm      yaw_arm    
##  Min.   :-623   Min.   :-180   Min.   :-89   Min.   :-180  
##  1st Qu.:-375   1st Qu.: -32   1st Qu.:-26   1st Qu.: -43  
##  Median :-320   Median :   0   Median :  0   Median :   0  
##  Mean   :-345   Mean   :  18   Mean   : -5   Mean   :  -1  
##  3rd Qu.:-306   3rd Qu.:  77   3rd Qu.: 11   3rd Qu.:  46  
##  Max.   : 293   Max.   : 180   Max.   : 88   Max.   : 180  
##                                                            
##  total_accel_arm var_accel_arm    avg_roll_arm   stddev_roll_arm
##  Min.   : 1      Min.   :  0     Min.   :-167    Min.   :  0    
##  1st Qu.:17      1st Qu.:  9     1st Qu.: -38    1st Qu.:  1    
##  Median :27      Median : 41     Median :   0    Median :  6    
##  Mean   :26      Mean   : 53     Mean   :  13    Mean   : 11    
##  3rd Qu.:33      3rd Qu.: 76     3rd Qu.:  76    3rd Qu.: 15    
##  Max.   :66      Max.   :332     Max.   : 163    Max.   :162    
##                  NA's   :19216   NA's   :19216   NA's   :19216  
##   var_roll_arm   avg_pitch_arm   stddev_pitch_arm var_pitch_arm  
##  Min.   :    0   Min.   :-82     Min.   : 0       Min.   :   0   
##  1st Qu.:    2   1st Qu.:-23     1st Qu.: 2       1st Qu.:   3   
##  Median :   33   Median :  0     Median : 8       Median :  66   
##  Mean   :  417   Mean   : -5     Mean   :10       Mean   : 196   
##  3rd Qu.:  223   3rd Qu.:  8     3rd Qu.:16       3rd Qu.: 267   
##  Max.   :26232   Max.   : 76     Max.   :43       Max.   :1885   
##  NA's   :19216   NA's   :19216   NA's   :19216    NA's   :19216  
##   avg_yaw_arm    stddev_yaw_arm   var_yaw_arm     gyros_arm_x  gyros_arm_y
##  Min.   :-173    Min.   :  0     Min.   :    0   Min.   :-6   Min.   :-3  
##  1st Qu.: -29    1st Qu.:  3     1st Qu.:    7   1st Qu.:-1   1st Qu.:-1  
##  Median :   0    Median : 17     Median :  278   Median : 0   Median : 0  
##  Mean   :   2    Mean   : 22     Mean   : 1056   Mean   : 0   Mean   : 0  
##  3rd Qu.:  38    3rd Qu.: 36     3rd Qu.: 1295   3rd Qu.: 2   3rd Qu.: 0  
##  Max.   : 152    Max.   :177     Max.   :31345   Max.   : 5   Max.   : 3  
##  NA's   :19216   NA's   :19216   NA's   :19216                            
##   gyros_arm_z  accel_arm_x    accel_arm_y    accel_arm_z    magnet_arm_x 
##  Min.   :-2   Min.   :-404   Min.   :-318   Min.   :-636   Min.   :-584  
##  1st Qu.: 0   1st Qu.:-242   1st Qu.: -54   1st Qu.:-143   1st Qu.:-300  
##  Median : 0   Median : -44   Median :  14   Median : -47   Median : 289  
##  Mean   : 0   Mean   : -60   Mean   :  33   Mean   : -71   Mean   : 192  
##  3rd Qu.: 1   3rd Qu.:  84   3rd Qu.: 139   3rd Qu.:  23   3rd Qu.: 637  
##  Max.   : 3   Max.   : 437   Max.   : 308   Max.   : 292   Max.   : 782  
##                                                                          
##   magnet_arm_y   magnet_arm_z  kurtosis_roll_arm kurtosis_picth_arm
##  Min.   :-392   Min.   :-597   Min.   :-2        Min.   :-2        
##  1st Qu.:  -9   1st Qu.: 131   1st Qu.:-1        1st Qu.:-1        
##  Median : 202   Median : 444   Median :-1        Median :-1        
##  Mean   : 157   Mean   : 306   Mean   : 0        Mean   :-1        
##  3rd Qu.: 323   3rd Qu.: 545   3rd Qu.: 0        3rd Qu.: 0        
##  Max.   : 583   Max.   : 694   Max.   :21        Max.   :20        
##                                NA's   :19294     NA's   :19296     
##  kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
##  Min.   :-2       Min.   :-3        Min.   :-5         Min.   :-7      
##  1st Qu.:-1       1st Qu.:-1        1st Qu.:-1         1st Qu.:-1      
##  Median :-1       Median : 0        Median : 0         Median : 0      
##  Mean   : 0       Mean   : 0        Mean   : 0         Mean   : 0      
##  3rd Qu.: 0       3rd Qu.: 1        3rd Qu.: 0         3rd Qu.: 0      
##  Max.   :56       Max.   : 4        Max.   : 3         Max.   : 7      
##  NA's   :19227    NA's   :19293     NA's   :19296      NA's   :19227   
##   max_roll_arm   max_picth_arm    max_yaw_arm     min_roll_arm  
##  Min.   :-73     Min.   :-173    Min.   : 4      Min.   :-89    
##  1st Qu.:  0     1st Qu.:  -2    1st Qu.:29      1st Qu.:-42    
##  Median :  5     Median :  23    Median :34      Median :-22    
##  Mean   : 11     Mean   :  36    Mean   :35      Mean   :-21    
##  3rd Qu.: 27     3rd Qu.:  96    3rd Qu.:41      3rd Qu.:  0    
##  Max.   : 86     Max.   : 180    Max.   :65      Max.   : 66    
##  NA's   :19216   NA's   :19216   NA's   :19216   NA's   :19216  
##  min_pitch_arm    min_yaw_arm    amplitude_roll_arm amplitude_pitch_arm
##  Min.   :-180    Min.   : 1      Min.   :  0        Min.   :  0        
##  1st Qu.: -73    1st Qu.: 8      1st Qu.:  5        1st Qu.: 10        
##  Median : -34    Median :13      Median : 28        Median : 55        
##  Mean   : -34    Mean   :15      Mean   : 32        Mean   : 70        
##  3rd Qu.:   0    3rd Qu.:19      3rd Qu.: 51        3rd Qu.:115        
##  Max.   : 152    Max.   :38      Max.   :120        Max.   :360        
##  NA's   :19216   NA's   :19216   NA's   :19216      NA's   :19216      
##  amplitude_yaw_arm roll_dumbbell  pitch_dumbbell  yaw_dumbbell 
##  Min.   : 0        Min.   :-154   Min.   :-150   Min.   :-151  
##  1st Qu.:13        1st Qu.: -18   1st Qu.: -41   1st Qu.: -78  
##  Median :22        Median :  48   Median : -21   Median :  -3  
##  Mean   :21        Mean   :  24   Mean   : -11   Mean   :   2  
##  3rd Qu.:29        3rd Qu.:  68   3rd Qu.:  17   3rd Qu.:  80  
##  Max.   :52        Max.   : 154   Max.   : 149   Max.   : 155  
##  NA's   :19216                                                 
##  kurtosis_roll_dumbbell kurtosis_picth_dumbbell kurtosis_yaw_dumbbell
##  Min.   :-2             Min.   :-2              Mode:logical         
##  1st Qu.:-1             1st Qu.:-1              NA's:19622           
##  Median : 0             Median : 0                                   
##  Mean   : 0             Mean   : 0                                   
##  3rd Qu.: 1             3rd Qu.: 1                                   
##  Max.   :55             Max.   :56                                   
##  NA's   :19221          NA's   :19218                                
##  skewness_roll_dumbbell skewness_pitch_dumbbell skewness_yaw_dumbbell
##  Min.   :-7             Min.   :-7              Mode:logical         
##  1st Qu.:-1             1st Qu.:-1              NA's:19622           
##  Median : 0             Median : 0                                   
##  Mean   : 0             Mean   : 0                                   
##  3rd Qu.: 0             3rd Qu.: 1                                   
##  Max.   : 2             Max.   : 4                                   
##  NA's   :19220          NA's   :19217                                
##  max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
##  Min.   :-70       Min.   :-113       Min.   :-2       Min.   :-150     
##  1st Qu.:-27       1st Qu.: -67       1st Qu.:-1       1st Qu.: -60     
##  Median : 15       Median :  40       Median : 0       Median : -44     
##  Mean   : 14       Mean   :  33       Mean   : 0       Mean   : -41     
##  3rd Qu.: 51       3rd Qu.: 133       3rd Qu.: 1       3rd Qu.: -25     
##  Max.   :137       Max.   : 155       Max.   :55       Max.   :  73     
##  NA's   :19216     NA's   :19216      NA's   :19221    NA's   :19216    
##  min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
##  Min.   :-147       Min.   :-2       Min.   :  0            
##  1st Qu.: -92       1st Qu.:-1       1st Qu.: 15            
##  Median : -66       Median : 0       Median : 35            
##  Mean   : -33       Mean   : 0       Mean   : 55            
##  3rd Qu.:  21       3rd Qu.: 1       3rd Qu.: 81            
##  Max.   : 121       Max.   :55       Max.   :256            
##  NA's   :19216      NA's   :19221    NA's   :19216          
##  amplitude_pitch_dumbbell amplitude_yaw_dumbbell total_accel_dumbbell
##  Min.   :  0              Min.   :0              Min.   : 0          
##  1st Qu.: 17              1st Qu.:0              1st Qu.: 4          
##  Median : 42              Median :0              Median :10          
##  Mean   : 66              Mean   :0              Mean   :14          
##  3rd Qu.:100              3rd Qu.:0              3rd Qu.:19          
##  Max.   :274              Max.   :0              Max.   :58          
##  NA's   :19216            NA's   :19221                              
##  var_accel_dumbbell avg_roll_dumbbell stddev_roll_dumbbell
##  Min.   :  0        Min.   :-129      Min.   :  0         
##  1st Qu.:  0        1st Qu.: -12      1st Qu.:  5         
##  Median :  1        Median :  48      Median : 12         
##  Mean   :  4        Mean   :  24      Mean   : 21         
##  3rd Qu.:  3        3rd Qu.:  64      3rd Qu.: 26         
##  Max.   :230        Max.   : 126      Max.   :124         
##  NA's   :19216      NA's   :19216     NA's   :19216       
##  var_roll_dumbbell avg_pitch_dumbbell stddev_pitch_dumbbell
##  Min.   :    0     Min.   :-71        Min.   : 0           
##  1st Qu.:   22     1st Qu.:-42        1st Qu.: 3           
##  Median :  149     Median :-20        Median : 8           
##  Mean   : 1020     Mean   :-12        Mean   :13           
##  3rd Qu.:  695     3rd Qu.: 13        3rd Qu.:19           
##  Max.   :15321     Max.   : 94        Max.   :83           
##  NA's   :19216     NA's   :19216      NA's   :19216        
##  var_pitch_dumbbell avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell
##  Min.   :   0       Min.   :-118     Min.   :  0         Min.   :    0   
##  1st Qu.:  12       1st Qu.: -77     1st Qu.:  4         1st Qu.:   15   
##  Median :  65       Median :  -5     Median : 10         Median :  105   
##  Mean   : 350       Mean   :   0     Mean   : 17         Mean   :  590   
##  3rd Qu.: 370       3rd Qu.:  71     3rd Qu.: 25         3rd Qu.:  609   
##  Max.   :6836       Max.   : 135     Max.   :107         Max.   :11468   
##  NA's   :19216      NA's   :19216    NA's   :19216       NA's   :19216   
##  gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x
##  Min.   :-204     Min.   :-2       Min.   : -2      Min.   :-419    
##  1st Qu.:   0     1st Qu.: 0       1st Qu.:  0      1st Qu.: -50    
##  Median :   0     Median : 0       Median :  0      Median :  -8    
##  Mean   :   0     Mean   : 0       Mean   :  0      Mean   : -29    
##  3rd Qu.:   0     3rd Qu.: 0       3rd Qu.:  0      3rd Qu.:  11    
##  Max.   :   2     Max.   :52       Max.   :317      Max.   : 235    
##                                                                     
##  accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
##  Min.   :-189     Min.   :-334     Min.   :-643      Min.   :-3600    
##  1st Qu.:  -8     1st Qu.:-142     1st Qu.:-535      1st Qu.:  231    
##  Median :  42     Median :  -1     Median :-479      Median :  311    
##  Mean   :  53     Mean   : -38     Mean   :-328      Mean   :  221    
##  3rd Qu.: 111     3rd Qu.:  38     3rd Qu.:-304      3rd Qu.:  390    
##  Max.   : 315     Max.   : 318     Max.   : 592      Max.   :  633    
##                                                                       
##  magnet_dumbbell_z  roll_forearm  pitch_forearm  yaw_forearm  
##  Min.   :-262      Min.   :-180   Min.   :-72   Min.   :-180  
##  1st Qu.: -45      1st Qu.:  -1   1st Qu.:  0   1st Qu.: -69  
##  Median :  13      Median :  22   Median :  9   Median :   0  
##  Mean   :  46      Mean   :  34   Mean   : 11   Mean   :  19  
##  3rd Qu.:  95      3rd Qu.: 140   3rd Qu.: 28   3rd Qu.: 110  
##  Max.   : 452      Max.   : 180   Max.   : 90   Max.   : 180  
##                                                               
##  kurtosis_roll_forearm kurtosis_picth_forearm kurtosis_yaw_forearm
##  Min.   :-2            Min.   :-2             Mode:logical        
##  1st Qu.:-1            1st Qu.:-1             NA's:19622          
##  Median :-1            Median :-1                                 
##  Mean   :-1            Mean   : 0                                 
##  3rd Qu.:-1            3rd Qu.: 0                                 
##  Max.   :40            Max.   :34                                 
##  NA's   :19300         NA's   :19301                              
##  skewness_roll_forearm skewness_pitch_forearm skewness_yaw_forearm
##  Min.   :-2            Min.   :-5             Mode:logical        
##  1st Qu.: 0            1st Qu.:-1             NA's:19622          
##  Median : 0            Median : 0                                 
##  Mean   : 0            Mean   : 0                                 
##  3rd Qu.: 0            3rd Qu.: 1                                 
##  Max.   : 6            Max.   : 4                                 
##  NA's   :19299         NA's   :19301                              
##  max_roll_forearm max_picth_forearm max_yaw_forearm min_roll_forearm
##  Min.   :-67      Min.   :-151      Min.   :-2      Min.   :-72     
##  1st Qu.:  0      1st Qu.:   0      1st Qu.:-1      1st Qu.: -6     
##  Median : 27      Median : 113      Median :-1      Median :  0     
##  Mean   : 24      Mean   :  81      Mean   :-1      Mean   :  0     
##  3rd Qu.: 46      3rd Qu.: 175      3rd Qu.:-1      3rd Qu.: 12     
##  Max.   : 90      Max.   : 180      Max.   :40      Max.   : 62     
##  NA's   :19216    NA's   :19216     NA's   :19300   NA's   :19216   
##  min_pitch_forearm min_yaw_forearm amplitude_roll_forearm
##  Min.   :-180      Min.   :-2      Min.   :  0           
##  1st Qu.:-175      1st Qu.:-1      1st Qu.:  1           
##  Median : -61      Median :-1      Median : 18           
##  Mean   : -58      Mean   :-1      Mean   : 25           
##  3rd Qu.:   0      3rd Qu.:-1      3rd Qu.: 40           
##  Max.   : 167      Max.   :40      Max.   :126           
##  NA's   :19216     NA's   :19300   NA's   :19216         
##  amplitude_pitch_forearm amplitude_yaw_forearm total_accel_forearm
##  Min.   :  0             Min.   :0             Min.   :  0        
##  1st Qu.:  2             1st Qu.:0             1st Qu.: 29        
##  Median : 84             Median :0             Median : 36        
##  Mean   :139             Mean   :0             Mean   : 35        
##  3rd Qu.:350             3rd Qu.:0             3rd Qu.: 41        
##  Max.   :360             Max.   :0             Max.   :108        
##  NA's   :19216           NA's   :19300                            
##  var_accel_forearm avg_roll_forearm stddev_roll_forearm var_roll_forearm
##  Min.   :  0       Min.   :-177     Min.   :  0         Min.   :    0   
##  1st Qu.:  7       1st Qu.:  -1     1st Qu.:  0         1st Qu.:    0   
##  Median : 21       Median :  11     Median :  8         Median :   64   
##  Mean   : 34       Mean   :  33     Mean   : 42         Mean   : 5274   
##  3rd Qu.: 51       3rd Qu.: 107     3rd Qu.: 85         3rd Qu.: 7289   
##  Max.   :173       Max.   : 177     Max.   :179         Max.   :32102   
##  NA's   :19216     NA's   :19216    NA's   :19216       NA's   :19216   
##  avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
##  Min.   :-68       Min.   : 0           Min.   :   0      Min.   :-155   
##  1st Qu.:  0       1st Qu.: 0           1st Qu.:   0      1st Qu.: -26   
##  Median : 12       Median : 6           Median :  30      Median :   0   
##  Mean   : 12       Mean   : 8           Mean   : 140      Mean   :  18   
##  3rd Qu.: 28       3rd Qu.:13           3rd Qu.: 166      3rd Qu.:  86   
##  Max.   : 72       Max.   :48           Max.   :2280      Max.   : 169   
##  NA's   :19216     NA's   :19216        NA's   :19216     NA's   :19216  
##  stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
##  Min.   :  0        Min.   :    0   Min.   :-22     Min.   : -7    
##  1st Qu.:  1        1st Qu.:    0   1st Qu.:  0     1st Qu.: -1    
##  Median : 25        Median :  612   Median :  0     Median :  0    
##  Mean   : 45        Mean   : 4640   Mean   :  0     Mean   :  0    
##  3rd Qu.: 86        3rd Qu.: 7368   3rd Qu.:  1     3rd Qu.:  2    
##  Max.   :198        Max.   :39009   Max.   :  4     Max.   :311    
##  NA's   :19216      NA's   :19216                                  
##  gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
##  Min.   : -8     Min.   :-498    Min.   :-632    Min.   :-446   
##  1st Qu.:  0     1st Qu.:-178    1st Qu.:  57    1st Qu.:-182   
##  Median :  0     Median : -57    Median : 201    Median : -39   
##  Mean   :  0     Mean   : -62    Mean   : 164    Mean   : -55   
##  3rd Qu.:  0     3rd Qu.:  76    3rd Qu.: 312    3rd Qu.:  26   
##  Max.   :231     Max.   : 477    Max.   : 923    Max.   : 291   
##                                                                 
##  magnet_forearm_x magnet_forearm_y magnet_forearm_z classe  
##  Min.   :-1280    Min.   :-896     Min.   :-973     A:5580  
##  1st Qu.: -616    1st Qu.:   2     1st Qu.: 191     B:3797  
##  Median : -378    Median : 591     Median : 511     C:3422  
##  Mean   : -313    Mean   : 380     Mean   : 394     D:3216  
##  3rd Qu.:  -73    3rd Qu.: 737     3rd Qu.: 653     E:3607  
##  Max.   :  672    Max.   :1480     Max.   :1090             
## 
```
Variables that are mosting missing and near zero variance are removed, along with several id-type variable. Data will be split into training and validation data frames for model building and evaluation

```r
#remove X variable and other ID variables
training2 <- training[,-c(1:7)]
ncol(training);ncol(training2)
```

```
## [1] 160
```

```
## [1] 153
```

```r
#remove columns with more than 50% NA values
training3=training2[, colSums(is.na(training2)) <= nrow(training2) * 0.5]
ncol(training2);ncol(training3)
```

```
## [1] 153
```

```
## [1] 53
```

```r
#remove columns with zero variance
nzv <-nearZeroVar(training3, saveMetrics= TRUE)
training4 <- training3[, nzv$nzv==FALSE]
ncol(training3);ncol(training4)
```

```
## [1] 53
```

```
## [1] 52
```

```r
str(training4)
```

```
## 'data.frame':	19622 obs. of  52 variables:
##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y        : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x        : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y        : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z        : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x       : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y       : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z       : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm           : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y         : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z         : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x         : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y         : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z         : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x        : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y        : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z        : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ roll_dumbbell       : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell      : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell        : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 0 0 -0.02 0 0 0 0 0 0 ...
##  $ accel_dumbbell_x    : int  -234 -233 -232 -232 -233 -234 -232 -234 -232 -235 ...
##  $ accel_dumbbell_y    : int  47 47 46 48 48 48 47 46 47 48 ...
##  $ accel_dumbbell_z    : int  -271 -269 -270 -269 -270 -269 -270 -272 -269 -270 ...
##  $ magnet_dumbbell_x   : int  -559 -555 -561 -552 -554 -558 -551 -555 -549 -558 ...
##  $ magnet_dumbbell_y   : int  293 296 298 303 292 294 295 300 292 291 ...
##  $ magnet_dumbbell_z   : num  -65 -64 -63 -60 -68 -66 -70 -74 -65 -69 ...
##  $ roll_forearm        : num  28.4 28.3 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 ...
##  $ yaw_forearm         : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -152 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
##  $ gyros_forearm_y     : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
##  $ gyros_forearm_z     : num  -0.02 -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 ...
##  $ accel_forearm_x     : int  192 192 196 189 189 193 195 193 193 190 ...
##  $ accel_forearm_y     : int  203 203 204 206 206 203 205 205 204 205 ...
##  $ accel_forearm_z     : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
##  $ magnet_forearm_x    : int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
##  $ magnet_forearm_y    : num  654 661 658 658 655 660 659 660 653 656 ...
##  $ magnet_forearm_z    : num  476 473 469 469 473 478 470 474 476 473 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
#create validation set
set.seed(67915)
inTrain = createDataPartition(training4$classe, p = 3/4)[[1]]
training5 = training4[ inTrain,]
validation = training4[-inTrain,]
```
Model Selection
-------------------------------------
Three model prediction methods will be compared: Decision Tree, Random Forest, and Boosting

```r
#Decision Tree
model1 <- train(classe~., data=training5, method="rpart")
varImp_tree <- varImp(model1)
varImp_tree
```

```
## rpart variable importance
## 
##   only 20 most important variables shown (out of 51)
## 
##                   Overall
## pitch_forearm       100.0
## roll_forearm         72.6
## roll_belt            72.5
## magnet_dumbbell_y    49.2
## accel_belt_z         44.5
## magnet_belt_y        42.7
## yaw_belt             42.1
## total_accel_belt     36.5
## magnet_arm_x         27.5
## accel_arm_x          26.4
## roll_dumbbell        19.0
## magnet_dumbbell_z    18.7
## magnet_dumbbell_x    18.0
## accel_dumbbell_y     16.3
## roll_arm             15.9
## gyros_belt_z          0.0
## gyros_belt_x          0.0
## accel_dumbbell_z      0.0
## magnet_belt_x         0.0
## total_accel_arm       0.0
```

```r
fancyRpartPlot(model1$finalModel)
```

![](projectv3_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

```r
#print(model1$finalModel)
predtree <- predict(model1,newdata =validation)
confusionMatrix(predtree,validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1284  392  388  380  141
##          B   21  331   23  126  137
##          C   88  226  444  298  236
##          D    0    0    0    0    0
##          E    2    0    0    0  387
## 
## Overall Statistics
##                                         
##                Accuracy : 0.499         
##                  95% CI : (0.485, 0.513)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.344         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.920   0.3488   0.5193    0.000   0.4295
## Specificity             0.629   0.9224   0.7906    1.000   0.9995
## Pos Pred Value          0.497   0.5188   0.3437      NaN   0.9949
## Neg Pred Value          0.952   0.8551   0.8862    0.836   0.8862
## Prevalence              0.284   0.1935   0.1743    0.164   0.1837
## Detection Rate          0.262   0.0675   0.0905    0.000   0.0789
## Detection Prevalence    0.527   0.1301   0.2635    0.000   0.0793
## Balanced Accuracy       0.775   0.6356   0.6549    0.500   0.7145
```

```r
##was model overfit?
tree <- predict(model1,newdata =training5)
confusionMatrix(tree,training5$classe)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##              0              0              0              1              0 
## AccuracyPValue  McnemarPValue 
##              0            NaN
```

```r
###no, accuracy equal on training and validation
```


```r
#Random Forest
library(randomForest)
model2 <- randomForest(as.factor(classe) ~ ., data=training5)
order(varImp(model2), decreasing = T)
```

```
##  [1]  1  3 40 38  2 37 39 36 34 26 10 13 12 35 14 46 23 51  7 29 31 11 28
## [24] 33 24 48 50  4 20 49 25 15 27 41 21 47 18 22 17 44 30  8  9 42 16  6
## [47]  5 32 45 43 19
```

```r
varImpPlot(model2,type=2)
```

![](projectv3_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

```r
##Reference:https://www.r-bloggers.com/variable-importance-plot-and-variable-selection/
predrf <- predict(model2,newdata =validation)
confusionMatrix(predrf,validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    4    0    0    0
##          B    0  944    5    0    0
##          C    0    1  850   10    0
##          D    0    0    0  794    1
##          E    0    0    0    0  900
## 
## Overall Statistics
##                                         
##                Accuracy : 0.996         
##                  95% CI : (0.993, 0.997)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.995         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.995    0.994    0.988    0.999
## Specificity             0.999    0.999    0.997    1.000    1.000
## Pos Pred Value          0.997    0.995    0.987    0.999    1.000
## Neg Pred Value          1.000    0.999    0.999    0.998    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.173    0.162    0.184
## Detection Prevalence    0.285    0.194    0.176    0.162    0.184
## Balanced Accuracy       0.999    0.997    0.996    0.994    0.999
```

```r
confusionMatrix(predrf,validation$classe)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##              1              1              1              1              0 
## AccuracyPValue  McnemarPValue 
##              0            NaN
```

```r
##plot top 15 variables
importanceOrder=order(-model2$importance)
names=rownames(model2$importance)[importanceOrder][1:15]
#Random Forest on reduced number of variables
training6=subset(training5,select=c("classe",names))
model3 <- randomForest(classe~., data=training6)
varImpPlot(model3,type=2)
```

![](projectv3_files/figure-html/unnamed-chunk-4-2.png)<!-- -->

```r
predrf2 <- predict(model3,newdata =validation)
confusionMatrix(predrf2,validation$classe)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##              1              1              1              1              0 
## AccuracyPValue  McnemarPValue 
##              0            NaN
```


```r
#Boosting - creating a smaller data frame to run procedure
inTrain2 = createDataPartition(training5$classe, p = .3)[[1]]
training7 = training5[ inTrain2,]
model4 <- train(classe~., data = training7, method = "gbm")
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1352
##      2        1.5210             nan     0.1000    0.0946
##      3        1.4585             nan     0.1000    0.0646
##      4        1.4136             nan     0.1000    0.0578
##      5        1.3746             nan     0.1000    0.0478
##      6        1.3419             nan     0.1000    0.0492
##      7        1.3092             nan     0.1000    0.0391
##      8        1.2819             nan     0.1000    0.0323
##      9        1.2599             nan     0.1000    0.0399
##     10        1.2317             nan     0.1000    0.0274
##     20        1.0661             nan     0.1000    0.0163
##     40        0.8868             nan     0.1000    0.0075
##     60        0.7747             nan     0.1000    0.0058
##     80        0.6910             nan     0.1000    0.0059
##    100        0.6239             nan     0.1000    0.0027
##    120        0.5712             nan     0.1000    0.0024
##    140        0.5224             nan     0.1000    0.0028
##    150        0.5022             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1916
##      2        1.4834             nan     0.1000    0.1341
##      3        1.3944             nan     0.1000    0.1040
##      4        1.3261             nan     0.1000    0.0916
##      5        1.2657             nan     0.1000    0.0812
##      6        1.2125             nan     0.1000    0.0606
##      7        1.1690             nan     0.1000    0.0574
##      8        1.1318             nan     0.1000    0.0602
##      9        1.0935             nan     0.1000    0.0528
##     10        1.0588             nan     0.1000    0.0313
##     20        0.8529             nan     0.1000    0.0185
##     40        0.6359             nan     0.1000    0.0114
##     60        0.5047             nan     0.1000    0.0063
##     80        0.4127             nan     0.1000    0.0029
##    100        0.3458             nan     0.1000    0.0032
##    120        0.2927             nan     0.1000    0.0021
##    140        0.2506             nan     0.1000    0.0022
##    150        0.2322             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2280
##      2        1.4570             nan     0.1000    0.1766
##      3        1.3480             nan     0.1000    0.1200
##      4        1.2669             nan     0.1000    0.1133
##      5        1.1936             nan     0.1000    0.0837
##      6        1.1379             nan     0.1000    0.0790
##      7        1.0854             nan     0.1000    0.0618
##      8        1.0417             nan     0.1000    0.0699
##      9        0.9966             nan     0.1000    0.0523
##     10        0.9600             nan     0.1000    0.0564
##     20        0.7099             nan     0.1000    0.0212
##     40        0.4702             nan     0.1000    0.0086
##     60        0.3482             nan     0.1000    0.0044
##     80        0.2688             nan     0.1000    0.0025
##    100        0.2112             nan     0.1000    0.0038
##    120        0.1710             nan     0.1000    0.0010
##    140        0.1397             nan     0.1000    0.0016
##    150        0.1264             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1332
##      2        1.5209             nan     0.1000    0.0908
##      3        1.4597             nan     0.1000    0.0659
##      4        1.4145             nan     0.1000    0.0624
##      5        1.3744             nan     0.1000    0.0470
##      6        1.3429             nan     0.1000    0.0405
##      7        1.3151             nan     0.1000    0.0332
##      8        1.2905             nan     0.1000    0.0407
##      9        1.2650             nan     0.1000    0.0332
##     10        1.2412             nan     0.1000    0.0296
##     20        1.0778             nan     0.1000    0.0168
##     40        0.8971             nan     0.1000    0.0076
##     60        0.7818             nan     0.1000    0.0072
##     80        0.6992             nan     0.1000    0.0043
##    100        0.6313             nan     0.1000    0.0033
##    120        0.5763             nan     0.1000    0.0019
##    140        0.5265             nan     0.1000    0.0014
##    150        0.5055             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1901
##      2        1.4831             nan     0.1000    0.1217
##      3        1.3983             nan     0.1000    0.1061
##      4        1.3275             nan     0.1000    0.0923
##      5        1.2664             nan     0.1000    0.0683
##      6        1.2216             nan     0.1000    0.0581
##      7        1.1804             nan     0.1000    0.0567
##      8        1.1414             nan     0.1000    0.0500
##      9        1.1069             nan     0.1000    0.0460
##     10        1.0760             nan     0.1000    0.0480
##     20        0.8591             nan     0.1000    0.0211
##     40        0.6368             nan     0.1000    0.0097
##     60        0.5046             nan     0.1000    0.0070
##     80        0.4093             nan     0.1000    0.0028
##    100        0.3450             nan     0.1000    0.0021
##    120        0.2940             nan     0.1000    0.0034
##    140        0.2532             nan     0.1000    0.0013
##    150        0.2331             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2332
##      2        1.4556             nan     0.1000    0.1726
##      3        1.3455             nan     0.1000    0.1290
##      4        1.2628             nan     0.1000    0.0995
##      5        1.1978             nan     0.1000    0.0801
##      6        1.1438             nan     0.1000    0.0760
##      7        1.0938             nan     0.1000    0.0706
##      8        1.0491             nan     0.1000    0.0655
##      9        1.0076             nan     0.1000    0.0652
##     10        0.9658             nan     0.1000    0.0539
##     20        0.7131             nan     0.1000    0.0171
##     40        0.4811             nan     0.1000    0.0120
##     60        0.3539             nan     0.1000    0.0079
##     80        0.2688             nan     0.1000    0.0019
##    100        0.2142             nan     0.1000    0.0038
##    120        0.1722             nan     0.1000    0.0008
##    140        0.1407             nan     0.1000    0.0015
##    150        0.1288             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1329
##      2        1.5215             nan     0.1000    0.0853
##      3        1.4634             nan     0.1000    0.0710
##      4        1.4161             nan     0.1000    0.0632
##      5        1.3747             nan     0.1000    0.0473
##      6        1.3431             nan     0.1000    0.0414
##      7        1.3131             nan     0.1000    0.0378
##      8        1.2879             nan     0.1000    0.0330
##      9        1.2636             nan     0.1000    0.0297
##     10        1.2427             nan     0.1000    0.0286
##     20        1.0802             nan     0.1000    0.0169
##     40        0.8991             nan     0.1000    0.0108
##     60        0.7838             nan     0.1000    0.0061
##     80        0.6989             nan     0.1000    0.0032
##    100        0.6337             nan     0.1000    0.0029
##    120        0.5768             nan     0.1000    0.0014
##    140        0.5315             nan     0.1000    0.0023
##    150        0.5104             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1948
##      2        1.4844             nan     0.1000    0.1342
##      3        1.3973             nan     0.1000    0.0998
##      4        1.3310             nan     0.1000    0.0912
##      5        1.2722             nan     0.1000    0.0716
##      6        1.2239             nan     0.1000    0.0574
##      7        1.1849             nan     0.1000    0.0554
##      8        1.1470             nan     0.1000    0.0485
##      9        1.1141             nan     0.1000    0.0487
##     10        1.0814             nan     0.1000    0.0462
##     20        0.8650             nan     0.1000    0.0155
##     40        0.6445             nan     0.1000    0.0068
##     60        0.5109             nan     0.1000    0.0072
##     80        0.4159             nan     0.1000    0.0047
##    100        0.3528             nan     0.1000    0.0022
##    120        0.3033             nan     0.1000    0.0038
##    140        0.2594             nan     0.1000    0.0009
##    150        0.2415             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2373
##      2        1.4518             nan     0.1000    0.1556
##      3        1.3495             nan     0.1000    0.1266
##      4        1.2685             nan     0.1000    0.1057
##      5        1.2002             nan     0.1000    0.0792
##      6        1.1471             nan     0.1000    0.0725
##      7        1.0987             nan     0.1000    0.0683
##      8        1.0530             nan     0.1000    0.0689
##      9        1.0080             nan     0.1000    0.0594
##     10        0.9684             nan     0.1000    0.0565
##     20        0.7238             nan     0.1000    0.0189
##     40        0.4819             nan     0.1000    0.0063
##     60        0.3546             nan     0.1000    0.0061
##     80        0.2726             nan     0.1000    0.0029
##    100        0.2173             nan     0.1000    0.0020
##    120        0.1770             nan     0.1000    0.0008
##    140        0.1458             nan     0.1000    0.0012
##    150        0.1320             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1412
##      2        1.5169             nan     0.1000    0.0918
##      3        1.4540             nan     0.1000    0.0655
##      4        1.4074             nan     0.1000    0.0560
##      5        1.3693             nan     0.1000    0.0624
##      6        1.3293             nan     0.1000    0.0453
##      7        1.3002             nan     0.1000    0.0334
##      8        1.2771             nan     0.1000    0.0358
##      9        1.2530             nan     0.1000    0.0371
##     10        1.2296             nan     0.1000    0.0378
##     20        1.0631             nan     0.1000    0.0157
##     40        0.8841             nan     0.1000    0.0095
##     60        0.7730             nan     0.1000    0.0048
##     80        0.6872             nan     0.1000    0.0040
##    100        0.6233             nan     0.1000    0.0038
##    120        0.5692             nan     0.1000    0.0020
##    140        0.5218             nan     0.1000    0.0010
##    150        0.5023             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2048
##      2        1.4780             nan     0.1000    0.1382
##      3        1.3883             nan     0.1000    0.0983
##      4        1.3230             nan     0.1000    0.0875
##      5        1.2668             nan     0.1000    0.0705
##      6        1.2202             nan     0.1000    0.0712
##      7        1.1738             nan     0.1000    0.0654
##      8        1.1306             nan     0.1000    0.0510
##      9        1.0966             nan     0.1000    0.0394
##     10        1.0690             nan     0.1000    0.0453
##     20        0.8511             nan     0.1000    0.0238
##     40        0.6312             nan     0.1000    0.0092
##     60        0.5061             nan     0.1000    0.0066
##     80        0.4128             nan     0.1000    0.0025
##    100        0.3452             nan     0.1000    0.0043
##    120        0.2932             nan     0.1000    0.0037
##    140        0.2517             nan     0.1000    0.0013
##    150        0.2319             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2440
##      2        1.4560             nan     0.1000    0.1646
##      3        1.3533             nan     0.1000    0.1263
##      4        1.2722             nan     0.1000    0.1081
##      5        1.2016             nan     0.1000    0.0806
##      6        1.1463             nan     0.1000    0.0774
##      7        1.0945             nan     0.1000    0.0720
##      8        1.0443             nan     0.1000    0.0773
##      9        0.9941             nan     0.1000    0.0584
##     10        0.9558             nan     0.1000    0.0476
##     20        0.7107             nan     0.1000    0.0249
##     40        0.4816             nan     0.1000    0.0080
##     60        0.3544             nan     0.1000    0.0051
##     80        0.2706             nan     0.1000    0.0026
##    100        0.2125             nan     0.1000    0.0015
##    120        0.1709             nan     0.1000    0.0011
##    140        0.1391             nan     0.1000    0.0014
##    150        0.1254             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1339
##      2        1.5187             nan     0.1000    0.0909
##      3        1.4585             nan     0.1000    0.0695
##      4        1.4121             nan     0.1000    0.0514
##      5        1.3754             nan     0.1000    0.0531
##      6        1.3423             nan     0.1000    0.0431
##      7        1.3131             nan     0.1000    0.0377
##      8        1.2881             nan     0.1000    0.0328
##      9        1.2649             nan     0.1000    0.0369
##     10        1.2399             nan     0.1000    0.0370
##     20        1.0725             nan     0.1000    0.0200
##     40        0.8903             nan     0.1000    0.0069
##     60        0.7714             nan     0.1000    0.0042
##     80        0.6890             nan     0.1000    0.0042
##    100        0.6231             nan     0.1000    0.0035
##    120        0.5711             nan     0.1000    0.0021
##    140        0.5255             nan     0.1000    0.0005
##    150        0.5062             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1932
##      2        1.4831             nan     0.1000    0.1397
##      3        1.3943             nan     0.1000    0.1002
##      4        1.3285             nan     0.1000    0.0825
##      5        1.2741             nan     0.1000    0.0753
##      6        1.2256             nan     0.1000    0.0704
##      7        1.1805             nan     0.1000    0.0588
##      8        1.1429             nan     0.1000    0.0638
##      9        1.1032             nan     0.1000    0.0402
##     10        1.0732             nan     0.1000    0.0384
##     20        0.8636             nan     0.1000    0.0216
##     40        0.6450             nan     0.1000    0.0117
##     60        0.5046             nan     0.1000    0.0066
##     80        0.4146             nan     0.1000    0.0035
##    100        0.3477             nan     0.1000    0.0039
##    120        0.2942             nan     0.1000    0.0022
##    140        0.2517             nan     0.1000    0.0011
##    150        0.2344             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2382
##      2        1.4532             nan     0.1000    0.1714
##      3        1.3435             nan     0.1000    0.1162
##      4        1.2652             nan     0.1000    0.1024
##      5        1.1969             nan     0.1000    0.0819
##      6        1.1414             nan     0.1000    0.0801
##      7        1.0880             nan     0.1000    0.0718
##      8        1.0417             nan     0.1000    0.0612
##      9        1.0006             nan     0.1000    0.0493
##     10        0.9671             nan     0.1000    0.0498
##     20        0.7126             nan     0.1000    0.0209
##     40        0.4796             nan     0.1000    0.0123
##     60        0.3541             nan     0.1000    0.0083
##     80        0.2703             nan     0.1000    0.0033
##    100        0.2160             nan     0.1000    0.0020
##    120        0.1748             nan     0.1000    0.0023
##    140        0.1416             nan     0.1000    0.0016
##    150        0.1283             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1363
##      2        1.5133             nan     0.1000    0.0901
##      3        1.4514             nan     0.1000    0.0721
##      4        1.4032             nan     0.1000    0.0561
##      5        1.3645             nan     0.1000    0.0509
##      6        1.3283             nan     0.1000    0.0415
##      7        1.3001             nan     0.1000    0.0350
##      8        1.2746             nan     0.1000    0.0333
##      9        1.2521             nan     0.1000    0.0357
##     10        1.2277             nan     0.1000    0.0261
##     20        1.0674             nan     0.1000    0.0163
##     40        0.8893             nan     0.1000    0.0085
##     60        0.7743             nan     0.1000    0.0053
##     80        0.6923             nan     0.1000    0.0036
##    100        0.6268             nan     0.1000    0.0028
##    120        0.5716             nan     0.1000    0.0035
##    140        0.5278             nan     0.1000    0.0019
##    150        0.5077             nan     0.1000    0.0027
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1964
##      2        1.4792             nan     0.1000    0.1370
##      3        1.3903             nan     0.1000    0.1075
##      4        1.3210             nan     0.1000    0.0858
##      5        1.2652             nan     0.1000    0.0682
##      6        1.2172             nan     0.1000    0.0618
##      7        1.1735             nan     0.1000    0.0504
##      8        1.1377             nan     0.1000    0.0539
##      9        1.1027             nan     0.1000    0.0529
##     10        1.0701             nan     0.1000    0.0434
##     20        0.8597             nan     0.1000    0.0216
##     40        0.6349             nan     0.1000    0.0130
##     60        0.5067             nan     0.1000    0.0066
##     80        0.4192             nan     0.1000    0.0057
##    100        0.3475             nan     0.1000    0.0029
##    120        0.2976             nan     0.1000    0.0034
##    140        0.2522             nan     0.1000    0.0013
##    150        0.2337             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2364
##      2        1.4523             nan     0.1000    0.1704
##      3        1.3419             nan     0.1000    0.1224
##      4        1.2604             nan     0.1000    0.0990
##      5        1.1945             nan     0.1000    0.0882
##      6        1.1371             nan     0.1000    0.0834
##      7        1.0823             nan     0.1000    0.0671
##      8        1.0397             nan     0.1000    0.0563
##      9        1.0027             nan     0.1000    0.0546
##     10        0.9651             nan     0.1000    0.0559
##     20        0.7151             nan     0.1000    0.0214
##     40        0.4865             nan     0.1000    0.0110
##     60        0.3585             nan     0.1000    0.0053
##     80        0.2777             nan     0.1000    0.0042
##    100        0.2185             nan     0.1000    0.0027
##    120        0.1774             nan     0.1000    0.0019
##    140        0.1442             nan     0.1000    0.0008
##    150        0.1306             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1309
##      2        1.5195             nan     0.1000    0.0873
##      3        1.4608             nan     0.1000    0.0668
##      4        1.4172             nan     0.1000    0.0634
##      5        1.3757             nan     0.1000    0.0476
##      6        1.3433             nan     0.1000    0.0379
##      7        1.3165             nan     0.1000    0.0375
##      8        1.2902             nan     0.1000    0.0382
##      9        1.2637             nan     0.1000    0.0336
##     10        1.2414             nan     0.1000    0.0325
##     20        1.0805             nan     0.1000    0.0195
##     40        0.8975             nan     0.1000    0.0096
##     60        0.7796             nan     0.1000    0.0051
##     80        0.6942             nan     0.1000    0.0038
##    100        0.6282             nan     0.1000    0.0030
##    120        0.5744             nan     0.1000    0.0024
##    140        0.5288             nan     0.1000    0.0014
##    150        0.5081             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1934
##      2        1.4806             nan     0.1000    0.1291
##      3        1.3935             nan     0.1000    0.1048
##      4        1.3247             nan     0.1000    0.0787
##      5        1.2723             nan     0.1000    0.0721
##      6        1.2239             nan     0.1000    0.0587
##      7        1.1834             nan     0.1000    0.0610
##      8        1.1429             nan     0.1000    0.0632
##      9        1.1034             nan     0.1000    0.0454
##     10        1.0723             nan     0.1000    0.0478
##     20        0.8601             nan     0.1000    0.0185
##     40        0.6345             nan     0.1000    0.0100
##     60        0.5025             nan     0.1000    0.0054
##     80        0.4132             nan     0.1000    0.0046
##    100        0.3453             nan     0.1000    0.0010
##    120        0.2932             nan     0.1000    0.0022
##    140        0.2528             nan     0.1000    0.0001
##    150        0.2346             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2409
##      2        1.4560             nan     0.1000    0.1601
##      3        1.3484             nan     0.1000    0.1287
##      4        1.2655             nan     0.1000    0.1053
##      5        1.1966             nan     0.1000    0.0882
##      6        1.1381             nan     0.1000    0.0709
##      7        1.0895             nan     0.1000    0.0735
##      8        1.0411             nan     0.1000    0.0628
##      9        0.9994             nan     0.1000    0.0607
##     10        0.9594             nan     0.1000    0.0463
##     20        0.7165             nan     0.1000    0.0275
##     40        0.4773             nan     0.1000    0.0092
##     60        0.3491             nan     0.1000    0.0048
##     80        0.2740             nan     0.1000    0.0060
##    100        0.2166             nan     0.1000    0.0020
##    120        0.1750             nan     0.1000    0.0019
##    140        0.1443             nan     0.1000    0.0004
##    150        0.1314             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1264
##      2        1.5233             nan     0.1000    0.0861
##      3        1.4646             nan     0.1000    0.0660
##      4        1.4193             nan     0.1000    0.0542
##      5        1.3832             nan     0.1000    0.0497
##      6        1.3502             nan     0.1000    0.0431
##      7        1.3207             nan     0.1000    0.0379
##      8        1.2950             nan     0.1000    0.0333
##      9        1.2730             nan     0.1000    0.0316
##     10        1.2510             nan     0.1000    0.0316
##     20        1.0864             nan     0.1000    0.0176
##     40        0.9002             nan     0.1000    0.0104
##     60        0.7864             nan     0.1000    0.0034
##     80        0.7036             nan     0.1000    0.0042
##    100        0.6387             nan     0.1000    0.0015
##    120        0.5822             nan     0.1000    0.0029
##    140        0.5350             nan     0.1000    0.0010
##    150        0.5153             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1843
##      2        1.4869             nan     0.1000    0.1237
##      3        1.4011             nan     0.1000    0.0986
##      4        1.3344             nan     0.1000    0.0835
##      5        1.2787             nan     0.1000    0.0642
##      6        1.2352             nan     0.1000    0.0652
##      7        1.1917             nan     0.1000    0.0574
##      8        1.1531             nan     0.1000    0.0546
##      9        1.1182             nan     0.1000    0.0483
##     10        1.0848             nan     0.1000    0.0433
##     20        0.8717             nan     0.1000    0.0162
##     40        0.6528             nan     0.1000    0.0074
##     60        0.5124             nan     0.1000    0.0054
##     80        0.4215             nan     0.1000    0.0054
##    100        0.3538             nan     0.1000    0.0023
##    120        0.2998             nan     0.1000    0.0041
##    140        0.2569             nan     0.1000    0.0005
##    150        0.2394             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2350
##      2        1.4591             nan     0.1000    0.1682
##      3        1.3517             nan     0.1000    0.1204
##      4        1.2724             nan     0.1000    0.1030
##      5        1.2046             nan     0.1000    0.0866
##      6        1.1464             nan     0.1000    0.0720
##      7        1.0981             nan     0.1000    0.0708
##      8        1.0497             nan     0.1000    0.0692
##      9        1.0053             nan     0.1000    0.0552
##     10        0.9663             nan     0.1000    0.0476
##     20        0.7274             nan     0.1000    0.0262
##     40        0.4882             nan     0.1000    0.0087
##     60        0.3590             nan     0.1000    0.0088
##     80        0.2760             nan     0.1000    0.0040
##    100        0.2197             nan     0.1000    0.0017
##    120        0.1774             nan     0.1000    0.0019
##    140        0.1464             nan     0.1000    0.0006
##    150        0.1341             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1350
##      2        1.5173             nan     0.1000    0.0921
##      3        1.4559             nan     0.1000    0.0687
##      4        1.4092             nan     0.1000    0.0533
##      5        1.3724             nan     0.1000    0.0574
##      6        1.3352             nan     0.1000    0.0423
##      7        1.3083             nan     0.1000    0.0400
##      8        1.2827             nan     0.1000    0.0383
##      9        1.2584             nan     0.1000    0.0351
##     10        1.2334             nan     0.1000    0.0282
##     20        1.0696             nan     0.1000    0.0182
##     40        0.8833             nan     0.1000    0.0089
##     60        0.7697             nan     0.1000    0.0043
##     80        0.6854             nan     0.1000    0.0027
##    100        0.6197             nan     0.1000    0.0017
##    120        0.5625             nan     0.1000    0.0027
##    140        0.5151             nan     0.1000    0.0018
##    150        0.4948             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2022
##      2        1.4800             nan     0.1000    0.1408
##      3        1.3927             nan     0.1000    0.1058
##      4        1.3235             nan     0.1000    0.0910
##      5        1.2643             nan     0.1000    0.0725
##      6        1.2151             nan     0.1000    0.0601
##      7        1.1743             nan     0.1000    0.0534
##      8        1.1385             nan     0.1000    0.0562
##      9        1.0998             nan     0.1000    0.0465
##     10        1.0678             nan     0.1000    0.0376
##     20        0.8479             nan     0.1000    0.0191
##     40        0.6296             nan     0.1000    0.0116
##     60        0.4994             nan     0.1000    0.0057
##     80        0.4071             nan     0.1000    0.0064
##    100        0.3397             nan     0.1000    0.0046
##    120        0.2862             nan     0.1000    0.0043
##    140        0.2449             nan     0.1000    0.0029
##    150        0.2274             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2484
##      2        1.4508             nan     0.1000    0.1672
##      3        1.3429             nan     0.1000    0.1276
##      4        1.2607             nan     0.1000    0.0994
##      5        1.1941             nan     0.1000    0.0916
##      6        1.1346             nan     0.1000    0.0837
##      7        1.0780             nan     0.1000    0.0741
##      8        1.0304             nan     0.1000    0.0596
##      9        0.9888             nan     0.1000    0.0541
##     10        0.9514             nan     0.1000    0.0447
##     20        0.7115             nan     0.1000    0.0254
##     40        0.4767             nan     0.1000    0.0158
##     60        0.3426             nan     0.1000    0.0061
##     80        0.2618             nan     0.1000    0.0011
##    100        0.2064             nan     0.1000    0.0027
##    120        0.1655             nan     0.1000    0.0011
##    140        0.1342             nan     0.1000    0.0006
##    150        0.1217             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1390
##      2        1.5215             nan     0.1000    0.0911
##      3        1.4619             nan     0.1000    0.0626
##      4        1.4179             nan     0.1000    0.0673
##      5        1.3758             nan     0.1000    0.0451
##      6        1.3455             nan     0.1000    0.0454
##      7        1.3152             nan     0.1000    0.0395
##      8        1.2887             nan     0.1000    0.0362
##      9        1.2628             nan     0.1000    0.0331
##     10        1.2412             nan     0.1000    0.0287
##     20        1.0779             nan     0.1000    0.0186
##     40        0.8954             nan     0.1000    0.0081
##     60        0.7813             nan     0.1000    0.0060
##     80        0.6992             nan     0.1000    0.0034
##    100        0.6328             nan     0.1000    0.0037
##    120        0.5760             nan     0.1000    0.0035
##    140        0.5308             nan     0.1000    0.0016
##    150        0.5100             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1948
##      2        1.4789             nan     0.1000    0.1343
##      3        1.3937             nan     0.1000    0.1031
##      4        1.3248             nan     0.1000    0.0780
##      5        1.2729             nan     0.1000    0.0858
##      6        1.2195             nan     0.1000    0.0655
##      7        1.1753             nan     0.1000    0.0549
##      8        1.1388             nan     0.1000    0.0549
##      9        1.1028             nan     0.1000    0.0440
##     10        1.0705             nan     0.1000    0.0470
##     20        0.8632             nan     0.1000    0.0176
##     40        0.6414             nan     0.1000    0.0078
##     60        0.5041             nan     0.1000    0.0062
##     80        0.4141             nan     0.1000    0.0043
##    100        0.3469             nan     0.1000    0.0017
##    120        0.2992             nan     0.1000    0.0032
##    140        0.2571             nan     0.1000    0.0020
##    150        0.2402             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2390
##      2        1.4550             nan     0.1000    0.1646
##      3        1.3463             nan     0.1000    0.1189
##      4        1.2680             nan     0.1000    0.1028
##      5        1.1987             nan     0.1000    0.0946
##      6        1.1376             nan     0.1000    0.0875
##      7        1.0816             nan     0.1000    0.0784
##      8        1.0285             nan     0.1000    0.0555
##      9        0.9919             nan     0.1000    0.0525
##     10        0.9565             nan     0.1000    0.0510
##     20        0.7102             nan     0.1000    0.0221
##     40        0.4758             nan     0.1000    0.0116
##     60        0.3552             nan     0.1000    0.0086
##     80        0.2746             nan     0.1000    0.0021
##    100        0.2202             nan     0.1000    0.0015
##    120        0.1780             nan     0.1000    0.0023
##    140        0.1457             nan     0.1000    0.0010
##    150        0.1329             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1358
##      2        1.5179             nan     0.1000    0.0880
##      3        1.4566             nan     0.1000    0.0675
##      4        1.4120             nan     0.1000    0.0518
##      5        1.3765             nan     0.1000    0.0468
##      6        1.3462             nan     0.1000    0.0393
##      7        1.3192             nan     0.1000    0.0392
##      8        1.2924             nan     0.1000    0.0373
##      9        1.2688             nan     0.1000    0.0306
##     10        1.2457             nan     0.1000    0.0336
##     20        1.0791             nan     0.1000    0.0170
##     40        0.8975             nan     0.1000    0.0069
##     60        0.7823             nan     0.1000    0.0043
##     80        0.7002             nan     0.1000    0.0037
##    100        0.6346             nan     0.1000    0.0033
##    120        0.5797             nan     0.1000    0.0028
##    140        0.5357             nan     0.1000    0.0020
##    150        0.5148             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1853
##      2        1.4824             nan     0.1000    0.1251
##      3        1.3975             nan     0.1000    0.0976
##      4        1.3315             nan     0.1000    0.0856
##      5        1.2756             nan     0.1000    0.0726
##      6        1.2285             nan     0.1000    0.0646
##      7        1.1850             nan     0.1000    0.0598
##      8        1.1450             nan     0.1000    0.0541
##      9        1.1091             nan     0.1000    0.0475
##     10        1.0772             nan     0.1000    0.0446
##     20        0.8677             nan     0.1000    0.0190
##     40        0.6389             nan     0.1000    0.0053
##     60        0.5137             nan     0.1000    0.0054
##     80        0.4220             nan     0.1000    0.0039
##    100        0.3543             nan     0.1000    0.0044
##    120        0.3003             nan     0.1000    0.0023
##    140        0.2594             nan     0.1000    0.0009
##    150        0.2422             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2390
##      2        1.4578             nan     0.1000    0.1656
##      3        1.3487             nan     0.1000    0.1205
##      4        1.2703             nan     0.1000    0.1014
##      5        1.2049             nan     0.1000    0.0955
##      6        1.1427             nan     0.1000    0.0834
##      7        1.0892             nan     0.1000    0.0707
##      8        1.0450             nan     0.1000    0.0584
##      9        1.0044             nan     0.1000    0.0615
##     10        0.9621             nan     0.1000    0.0457
##     20        0.7225             nan     0.1000    0.0225
##     40        0.4897             nan     0.1000    0.0094
##     60        0.3646             nan     0.1000    0.0035
##     80        0.2857             nan     0.1000    0.0032
##    100        0.2254             nan     0.1000    0.0024
##    120        0.1830             nan     0.1000    0.0010
##    140        0.1518             nan     0.1000    0.0011
##    150        0.1376             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1346
##      2        1.5208             nan     0.1000    0.0926
##      3        1.4591             nan     0.1000    0.0663
##      4        1.4135             nan     0.1000    0.0564
##      5        1.3762             nan     0.1000    0.0505
##      6        1.3445             nan     0.1000    0.0424
##      7        1.3160             nan     0.1000    0.0434
##      8        1.2875             nan     0.1000    0.0341
##      9        1.2649             nan     0.1000    0.0377
##     10        1.2397             nan     0.1000    0.0307
##     20        1.0782             nan     0.1000    0.0167
##     40        0.9017             nan     0.1000    0.0055
##     60        0.7867             nan     0.1000    0.0055
##     80        0.7012             nan     0.1000    0.0038
##    100        0.6332             nan     0.1000    0.0041
##    120        0.5791             nan     0.1000    0.0024
##    140        0.5322             nan     0.1000    0.0012
##    150        0.5114             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1908
##      2        1.4805             nan     0.1000    0.1330
##      3        1.3937             nan     0.1000    0.0998
##      4        1.3284             nan     0.1000    0.0900
##      5        1.2714             nan     0.1000    0.0833
##      6        1.2180             nan     0.1000    0.0657
##      7        1.1759             nan     0.1000    0.0629
##      8        1.1331             nan     0.1000    0.0523
##      9        1.0983             nan     0.1000    0.0482
##     10        1.0661             nan     0.1000    0.0374
##     20        0.8632             nan     0.1000    0.0222
##     40        0.6507             nan     0.1000    0.0097
##     60        0.5153             nan     0.1000    0.0064
##     80        0.4175             nan     0.1000    0.0044
##    100        0.3515             nan     0.1000    0.0034
##    120        0.2974             nan     0.1000    0.0012
##    140        0.2557             nan     0.1000    0.0026
##    150        0.2369             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2398
##      2        1.4536             nan     0.1000    0.1798
##      3        1.3380             nan     0.1000    0.1185
##      4        1.2581             nan     0.1000    0.0942
##      5        1.1940             nan     0.1000    0.1008
##      6        1.1315             nan     0.1000    0.0720
##      7        1.0833             nan     0.1000    0.0762
##      8        1.0342             nan     0.1000    0.0553
##      9        0.9948             nan     0.1000    0.0510
##     10        0.9578             nan     0.1000    0.0509
##     20        0.7198             nan     0.1000    0.0249
##     40        0.4768             nan     0.1000    0.0092
##     60        0.3504             nan     0.1000    0.0039
##     80        0.2705             nan     0.1000    0.0029
##    100        0.2135             nan     0.1000    0.0018
##    120        0.1730             nan     0.1000    0.0019
##    140        0.1404             nan     0.1000    0.0002
##    150        0.1269             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1300
##      2        1.5214             nan     0.1000    0.0851
##      3        1.4631             nan     0.1000    0.0675
##      4        1.4163             nan     0.1000    0.0594
##      5        1.3737             nan     0.1000    0.0434
##      6        1.3434             nan     0.1000    0.0409
##      7        1.3148             nan     0.1000    0.0414
##      8        1.2888             nan     0.1000    0.0407
##      9        1.2604             nan     0.1000    0.0320
##     10        1.2376             nan     0.1000    0.0299
##     20        1.0775             nan     0.1000    0.0167
##     40        0.8940             nan     0.1000    0.0085
##     60        0.7785             nan     0.1000    0.0043
##     80        0.6971             nan     0.1000    0.0063
##    100        0.6312             nan     0.1000    0.0028
##    120        0.5750             nan     0.1000    0.0012
##    140        0.5281             nan     0.1000    0.0018
##    150        0.5074             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2000
##      2        1.4783             nan     0.1000    0.1363
##      3        1.3892             nan     0.1000    0.0981
##      4        1.3227             nan     0.1000    0.0795
##      5        1.2711             nan     0.1000    0.0838
##      6        1.2163             nan     0.1000    0.0697
##      7        1.1690             nan     0.1000    0.0507
##      8        1.1343             nan     0.1000    0.0455
##      9        1.1026             nan     0.1000    0.0477
##     10        1.0696             nan     0.1000    0.0382
##     20        0.8601             nan     0.1000    0.0236
##     40        0.6396             nan     0.1000    0.0120
##     60        0.5086             nan     0.1000    0.0089
##     80        0.4157             nan     0.1000    0.0023
##    100        0.3540             nan     0.1000    0.0023
##    120        0.3023             nan     0.1000    0.0027
##    140        0.2574             nan     0.1000    0.0013
##    150        0.2389             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2352
##      2        1.4528             nan     0.1000    0.1668
##      3        1.3445             nan     0.1000    0.1241
##      4        1.2624             nan     0.1000    0.1013
##      5        1.1955             nan     0.1000    0.0798
##      6        1.1416             nan     0.1000    0.0793
##      7        1.0904             nan     0.1000    0.0630
##      8        1.0471             nan     0.1000    0.0614
##      9        1.0041             nan     0.1000    0.0499
##     10        0.9701             nan     0.1000    0.0555
##     20        0.7214             nan     0.1000    0.0220
##     40        0.4861             nan     0.1000    0.0070
##     60        0.3569             nan     0.1000    0.0073
##     80        0.2763             nan     0.1000    0.0044
##    100        0.2208             nan     0.1000    0.0020
##    120        0.1808             nan     0.1000    0.0016
##    140        0.1474             nan     0.1000    0.0003
##    150        0.1351             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1414
##      2        1.5175             nan     0.1000    0.0899
##      3        1.4556             nan     0.1000    0.0716
##      4        1.4077             nan     0.1000    0.0549
##      5        1.3709             nan     0.1000    0.0568
##      6        1.3345             nan     0.1000    0.0395
##      7        1.3062             nan     0.1000    0.0335
##      8        1.2823             nan     0.1000    0.0327
##      9        1.2599             nan     0.1000    0.0380
##     10        1.2327             nan     0.1000    0.0296
##     20        1.0700             nan     0.1000    0.0196
##     40        0.8905             nan     0.1000    0.0076
##     60        0.7803             nan     0.1000    0.0038
##     80        0.6988             nan     0.1000    0.0032
##    100        0.6338             nan     0.1000    0.0024
##    120        0.5794             nan     0.1000    0.0015
##    140        0.5351             nan     0.1000    0.0023
##    150        0.5139             nan     0.1000    0.0028
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1912
##      2        1.4808             nan     0.1000    0.1359
##      3        1.3916             nan     0.1000    0.1059
##      4        1.3187             nan     0.1000    0.0799
##      5        1.2625             nan     0.1000    0.0694
##      6        1.2152             nan     0.1000    0.0672
##      7        1.1708             nan     0.1000    0.0602
##      8        1.1312             nan     0.1000    0.0482
##      9        1.0977             nan     0.1000    0.0495
##     10        1.0646             nan     0.1000    0.0382
##     20        0.8548             nan     0.1000    0.0162
##     40        0.6502             nan     0.1000    0.0079
##     60        0.5124             nan     0.1000    0.0080
##     80        0.4257             nan     0.1000    0.0062
##    100        0.3538             nan     0.1000    0.0028
##    120        0.3016             nan     0.1000    0.0015
##    140        0.2581             nan     0.1000    0.0014
##    150        0.2390             nan     0.1000    0.0003
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2432
##      2        1.4522             nan     0.1000    0.1565
##      3        1.3462             nan     0.1000    0.1164
##      4        1.2672             nan     0.1000    0.0964
##      5        1.2023             nan     0.1000    0.0876
##      6        1.1456             nan     0.1000    0.0683
##      7        1.0975             nan     0.1000    0.0779
##      8        1.0470             nan     0.1000    0.0598
##      9        1.0064             nan     0.1000    0.0649
##     10        0.9644             nan     0.1000    0.0422
##     20        0.7165             nan     0.1000    0.0200
##     40        0.4895             nan     0.1000    0.0127
##     60        0.3612             nan     0.1000    0.0030
##     80        0.2787             nan     0.1000    0.0037
##    100        0.2209             nan     0.1000    0.0030
##    120        0.1769             nan     0.1000    0.0012
##    140        0.1446             nan     0.1000    0.0012
##    150        0.1306             nan     0.1000    0.0003
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1331
##      2        1.5194             nan     0.1000    0.0954
##      3        1.4591             nan     0.1000    0.0673
##      4        1.4140             nan     0.1000    0.0580
##      5        1.3755             nan     0.1000    0.0538
##      6        1.3401             nan     0.1000    0.0467
##      7        1.3085             nan     0.1000    0.0385
##      8        1.2810             nan     0.1000    0.0356
##      9        1.2557             nan     0.1000    0.0293
##     10        1.2348             nan     0.1000    0.0307
##     20        1.0708             nan     0.1000    0.0145
##     40        0.8889             nan     0.1000    0.0091
##     60        0.7740             nan     0.1000    0.0050
##     80        0.6901             nan     0.1000    0.0037
##    100        0.6256             nan     0.1000    0.0037
##    120        0.5728             nan     0.1000    0.0028
##    140        0.5298             nan     0.1000    0.0022
##    150        0.5101             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1997
##      2        1.4812             nan     0.1000    0.1357
##      3        1.3918             nan     0.1000    0.1012
##      4        1.3267             nan     0.1000    0.0843
##      5        1.2705             nan     0.1000    0.0738
##      6        1.2214             nan     0.1000    0.0670
##      7        1.1770             nan     0.1000    0.0606
##      8        1.1376             nan     0.1000    0.0572
##      9        1.1016             nan     0.1000    0.0524
##     10        1.0682             nan     0.1000    0.0398
##     20        0.8614             nan     0.1000    0.0168
##     40        0.6428             nan     0.1000    0.0091
##     60        0.5061             nan     0.1000    0.0057
##     80        0.4140             nan     0.1000    0.0028
##    100        0.3467             nan     0.1000    0.0022
##    120        0.2954             nan     0.1000    0.0024
##    140        0.2556             nan     0.1000    0.0013
##    150        0.2374             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2484
##      2        1.4481             nan     0.1000    0.1628
##      3        1.3439             nan     0.1000    0.1268
##      4        1.2618             nan     0.1000    0.0998
##      5        1.1954             nan     0.1000    0.1009
##      6        1.1296             nan     0.1000    0.0772
##      7        1.0791             nan     0.1000    0.0655
##      8        1.0360             nan     0.1000    0.0612
##      9        0.9964             nan     0.1000    0.0447
##     10        0.9648             nan     0.1000    0.0583
##     20        0.7181             nan     0.1000    0.0222
##     40        0.4801             nan     0.1000    0.0071
##     60        0.3526             nan     0.1000    0.0059
##     80        0.2737             nan     0.1000    0.0037
##    100        0.2145             nan     0.1000    0.0016
##    120        0.1740             nan     0.1000    0.0024
##    140        0.1419             nan     0.1000    0.0010
##    150        0.1298             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1366
##      2        1.5205             nan     0.1000    0.0874
##      3        1.4596             nan     0.1000    0.0645
##      4        1.4141             nan     0.1000    0.0591
##      5        1.3739             nan     0.1000    0.0485
##      6        1.3419             nan     0.1000    0.0400
##      7        1.3136             nan     0.1000    0.0391
##      8        1.2870             nan     0.1000    0.0337
##      9        1.2653             nan     0.1000    0.0329
##     10        1.2433             nan     0.1000    0.0292
##     20        1.0818             nan     0.1000    0.0227
##     40        0.8930             nan     0.1000    0.0090
##     60        0.7775             nan     0.1000    0.0038
##     80        0.6930             nan     0.1000    0.0040
##    100        0.6269             nan     0.1000    0.0027
##    120        0.5740             nan     0.1000    0.0028
##    140        0.5285             nan     0.1000    0.0015
##    150        0.5087             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1869
##      2        1.4839             nan     0.1000    0.1311
##      3        1.3935             nan     0.1000    0.1000
##      4        1.3276             nan     0.1000    0.0899
##      5        1.2669             nan     0.1000    0.0763
##      6        1.2174             nan     0.1000    0.0660
##      7        1.1746             nan     0.1000    0.0552
##      8        1.1373             nan     0.1000    0.0539
##      9        1.1022             nan     0.1000    0.0405
##     10        1.0736             nan     0.1000    0.0447
##     20        0.8592             nan     0.1000    0.0175
##     40        0.6362             nan     0.1000    0.0106
##     60        0.5111             nan     0.1000    0.0050
##     80        0.4201             nan     0.1000    0.0039
##    100        0.3545             nan     0.1000    0.0037
##    120        0.3006             nan     0.1000    0.0014
##    140        0.2580             nan     0.1000    0.0018
##    150        0.2420             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2419
##      2        1.4535             nan     0.1000    0.1648
##      3        1.3457             nan     0.1000    0.1196
##      4        1.2664             nan     0.1000    0.0990
##      5        1.1995             nan     0.1000    0.0890
##      6        1.1407             nan     0.1000    0.0785
##      7        1.0872             nan     0.1000    0.0603
##      8        1.0456             nan     0.1000    0.0539
##      9        1.0088             nan     0.1000    0.0596
##     10        0.9682             nan     0.1000    0.0557
##     20        0.7059             nan     0.1000    0.0220
##     40        0.4772             nan     0.1000    0.0102
##     60        0.3513             nan     0.1000    0.0031
##     80        0.2719             nan     0.1000    0.0024
##    100        0.2182             nan     0.1000    0.0006
##    120        0.1785             nan     0.1000    0.0014
##    140        0.1428             nan     0.1000    0.0014
##    150        0.1300             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1274
##      2        1.5212             nan     0.1000    0.0881
##      3        1.4595             nan     0.1000    0.0671
##      4        1.4135             nan     0.1000    0.0501
##      5        1.3780             nan     0.1000    0.0411
##      6        1.3493             nan     0.1000    0.0477
##      7        1.3176             nan     0.1000    0.0355
##      8        1.2935             nan     0.1000    0.0388
##      9        1.2653             nan     0.1000    0.0322
##     10        1.2437             nan     0.1000    0.0303
##     20        1.0786             nan     0.1000    0.0160
##     40        0.8981             nan     0.1000    0.0073
##     60        0.7853             nan     0.1000    0.0057
##     80        0.7009             nan     0.1000    0.0056
##    100        0.6348             nan     0.1000    0.0033
##    120        0.5814             nan     0.1000    0.0015
##    140        0.5366             nan     0.1000    0.0011
##    150        0.5177             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1765
##      2        1.4856             nan     0.1000    0.1380
##      3        1.3958             nan     0.1000    0.1089
##      4        1.3268             nan     0.1000    0.0869
##      5        1.2710             nan     0.1000    0.0703
##      6        1.2219             nan     0.1000    0.0641
##      7        1.1803             nan     0.1000    0.0597
##      8        1.1404             nan     0.1000    0.0574
##      9        1.1021             nan     0.1000    0.0456
##     10        1.0716             nan     0.1000    0.0495
##     20        0.8556             nan     0.1000    0.0195
##     40        0.6364             nan     0.1000    0.0070
##     60        0.5104             nan     0.1000    0.0034
##     80        0.4217             nan     0.1000    0.0058
##    100        0.3564             nan     0.1000    0.0027
##    120        0.3039             nan     0.1000    0.0021
##    140        0.2622             nan     0.1000    0.0009
##    150        0.2426             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2396
##      2        1.4529             nan     0.1000    0.1656
##      3        1.3450             nan     0.1000    0.1281
##      4        1.2608             nan     0.1000    0.1098
##      5        1.1901             nan     0.1000    0.0898
##      6        1.1318             nan     0.1000    0.0769
##      7        1.0801             nan     0.1000    0.0733
##      8        1.0325             nan     0.1000    0.0615
##      9        0.9913             nan     0.1000    0.0521
##     10        0.9548             nan     0.1000    0.0409
##     20        0.7185             nan     0.1000    0.0291
##     40        0.4855             nan     0.1000    0.0105
##     60        0.3578             nan     0.1000    0.0050
##     80        0.2771             nan     0.1000    0.0015
##    100        0.2238             nan     0.1000    0.0024
##    120        0.1810             nan     0.1000    0.0015
##    140        0.1487             nan     0.1000    0.0023
##    150        0.1354             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1301
##      2        1.5205             nan     0.1000    0.0938
##      3        1.4569             nan     0.1000    0.0744
##      4        1.4083             nan     0.1000    0.0593
##      5        1.3691             nan     0.1000    0.0466
##      6        1.3367             nan     0.1000    0.0449
##      7        1.3063             nan     0.1000    0.0435
##      8        1.2786             nan     0.1000    0.0392
##      9        1.2531             nan     0.1000    0.0329
##     10        1.2310             nan     0.1000    0.0267
##     20        1.0711             nan     0.1000    0.0155
##     40        0.8907             nan     0.1000    0.0095
##     60        0.7722             nan     0.1000    0.0071
##     80        0.6885             nan     0.1000    0.0032
##    100        0.6235             nan     0.1000    0.0023
##    120        0.5694             nan     0.1000    0.0028
##    140        0.5213             nan     0.1000    0.0015
##    150        0.5003             nan     0.1000    0.0027
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2004
##      2        1.4790             nan     0.1000    0.1299
##      3        1.3909             nan     0.1000    0.1069
##      4        1.3205             nan     0.1000    0.0873
##      5        1.2614             nan     0.1000    0.0778
##      6        1.2126             nan     0.1000    0.0674
##      7        1.1683             nan     0.1000    0.0556
##      8        1.1311             nan     0.1000    0.0490
##      9        1.0993             nan     0.1000    0.0470
##     10        1.0680             nan     0.1000    0.0432
##     20        0.8553             nan     0.1000    0.0215
##     40        0.6353             nan     0.1000    0.0073
##     60        0.5052             nan     0.1000    0.0059
##     80        0.4173             nan     0.1000    0.0046
##    100        0.3476             nan     0.1000    0.0043
##    120        0.2925             nan     0.1000    0.0039
##    140        0.2504             nan     0.1000    0.0014
##    150        0.2345             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2385
##      2        1.4531             nan     0.1000    0.1651
##      3        1.3483             nan     0.1000    0.1229
##      4        1.2705             nan     0.1000    0.1031
##      5        1.2012             nan     0.1000    0.0859
##      6        1.1438             nan     0.1000    0.0798
##      7        1.0903             nan     0.1000    0.0677
##      8        1.0445             nan     0.1000    0.0632
##      9        1.0024             nan     0.1000    0.0567
##     10        0.9651             nan     0.1000    0.0518
##     20        0.7205             nan     0.1000    0.0184
##     40        0.4837             nan     0.1000    0.0119
##     60        0.3563             nan     0.1000    0.0049
##     80        0.2735             nan     0.1000    0.0025
##    100        0.2189             nan     0.1000    0.0019
##    120        0.1746             nan     0.1000    0.0018
##    140        0.1417             nan     0.1000    0.0016
##    150        0.1287             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1271
##      2        1.5217             nan     0.1000    0.0885
##      3        1.4618             nan     0.1000    0.0674
##      4        1.4160             nan     0.1000    0.0534
##      5        1.3806             nan     0.1000    0.0587
##      6        1.3433             nan     0.1000    0.0396
##      7        1.3167             nan     0.1000    0.0432
##      8        1.2890             nan     0.1000    0.0353
##      9        1.2639             nan     0.1000    0.0341
##     10        1.2414             nan     0.1000    0.0335
##     20        1.0790             nan     0.1000    0.0179
##     40        0.8971             nan     0.1000    0.0086
##     60        0.7828             nan     0.1000    0.0052
##     80        0.6994             nan     0.1000    0.0045
##    100        0.6298             nan     0.1000    0.0026
##    120        0.5781             nan     0.1000    0.0027
##    140        0.5333             nan     0.1000    0.0023
##    150        0.5141             nan     0.1000    0.0024
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2021
##      2        1.4822             nan     0.1000    0.1337
##      3        1.3975             nan     0.1000    0.1027
##      4        1.3320             nan     0.1000    0.0792
##      5        1.2777             nan     0.1000    0.0753
##      6        1.2291             nan     0.1000    0.0650
##      7        1.1858             nan     0.1000    0.0665
##      8        1.1431             nan     0.1000    0.0558
##      9        1.1069             nan     0.1000    0.0438
##     10        1.0772             nan     0.1000    0.0370
##     20        0.8706             nan     0.1000    0.0204
##     40        0.6457             nan     0.1000    0.0090
##     60        0.5137             nan     0.1000    0.0060
##     80        0.4236             nan     0.1000    0.0053
##    100        0.3577             nan     0.1000    0.0052
##    120        0.3018             nan     0.1000    0.0014
##    140        0.2609             nan     0.1000    0.0022
##    150        0.2438             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2459
##      2        1.4548             nan     0.1000    0.1632
##      3        1.3524             nan     0.1000    0.1277
##      4        1.2699             nan     0.1000    0.0994
##      5        1.2056             nan     0.1000    0.0793
##      6        1.1503             nan     0.1000    0.0761
##      7        1.0993             nan     0.1000    0.0609
##      8        1.0578             nan     0.1000    0.0659
##      9        1.0140             nan     0.1000    0.0623
##     10        0.9733             nan     0.1000    0.0468
##     20        0.7323             nan     0.1000    0.0279
##     40        0.4958             nan     0.1000    0.0102
##     60        0.3691             nan     0.1000    0.0093
##     80        0.2861             nan     0.1000    0.0028
##    100        0.2286             nan     0.1000    0.0019
##    120        0.1864             nan     0.1000    0.0011
##    140        0.1540             nan     0.1000    0.0007
##    150        0.1407             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1331
##      2        1.5165             nan     0.1000    0.0911
##      3        1.4558             nan     0.1000    0.0684
##      4        1.4094             nan     0.1000    0.0571
##      5        1.3717             nan     0.1000    0.0456
##      6        1.3413             nan     0.1000    0.0462
##      7        1.3122             nan     0.1000    0.0421
##      8        1.2851             nan     0.1000    0.0328
##      9        1.2616             nan     0.1000    0.0309
##     10        1.2401             nan     0.1000    0.0293
##     20        1.0749             nan     0.1000    0.0169
##     40        0.8993             nan     0.1000    0.0071
##     60        0.7846             nan     0.1000    0.0051
##     80        0.7018             nan     0.1000    0.0048
##    100        0.6367             nan     0.1000    0.0025
##    120        0.5813             nan     0.1000    0.0025
##    140        0.5367             nan     0.1000    0.0021
##    150        0.5167             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1935
##      2        1.4852             nan     0.1000    0.1372
##      3        1.3933             nan     0.1000    0.0994
##      4        1.3265             nan     0.1000    0.0817
##      5        1.2720             nan     0.1000    0.0726
##      6        1.2238             nan     0.1000    0.0626
##      7        1.1817             nan     0.1000    0.0558
##      8        1.1436             nan     0.1000    0.0534
##      9        1.1077             nan     0.1000    0.0439
##     10        1.0781             nan     0.1000    0.0347
##     20        0.8695             nan     0.1000    0.0208
##     40        0.6500             nan     0.1000    0.0147
##     60        0.5124             nan     0.1000    0.0087
##     80        0.4227             nan     0.1000    0.0081
##    100        0.3508             nan     0.1000    0.0017
##    120        0.2973             nan     0.1000    0.0016
##    140        0.2543             nan     0.1000    0.0022
##    150        0.2356             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2325
##      2        1.4577             nan     0.1000    0.1615
##      3        1.3534             nan     0.1000    0.1232
##      4        1.2714             nan     0.1000    0.1088
##      5        1.2005             nan     0.1000    0.0903
##      6        1.1373             nan     0.1000    0.0744
##      7        1.0886             nan     0.1000    0.0636
##      8        1.0460             nan     0.1000    0.0568
##      9        1.0068             nan     0.1000    0.0550
##     10        0.9699             nan     0.1000    0.0527
##     20        0.7225             nan     0.1000    0.0211
##     40        0.4867             nan     0.1000    0.0110
##     60        0.3625             nan     0.1000    0.0049
##     80        0.2793             nan     0.1000    0.0047
##    100        0.2235             nan     0.1000    0.0020
##    120        0.1801             nan     0.1000    0.0012
##    140        0.1487             nan     0.1000    0.0011
##    150        0.1353             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1373
##      2        1.5180             nan     0.1000    0.0920
##      3        1.4563             nan     0.1000    0.0654
##      4        1.4110             nan     0.1000    0.0570
##      5        1.3727             nan     0.1000    0.0425
##      6        1.3426             nan     0.1000    0.0484
##      7        1.3115             nan     0.1000    0.0390
##      8        1.2838             nan     0.1000    0.0318
##      9        1.2625             nan     0.1000    0.0269
##     10        1.2430             nan     0.1000    0.0366
##     20        1.0748             nan     0.1000    0.0151
##     40        0.8905             nan     0.1000    0.0091
##     60        0.7776             nan     0.1000    0.0068
##     80        0.6964             nan     0.1000    0.0042
##    100        0.6291             nan     0.1000    0.0026
##    120        0.5753             nan     0.1000    0.0012
##    140        0.5319             nan     0.1000    0.0015
##    150        0.5107             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1908
##      2        1.4840             nan     0.1000    0.1332
##      3        1.3981             nan     0.1000    0.1036
##      4        1.3273             nan     0.1000    0.0813
##      5        1.2727             nan     0.1000    0.0746
##      6        1.2239             nan     0.1000    0.0659
##      7        1.1806             nan     0.1000    0.0543
##      8        1.1428             nan     0.1000    0.0517
##      9        1.1079             nan     0.1000    0.0543
##     10        1.0711             nan     0.1000    0.0393
##     20        0.8667             nan     0.1000    0.0178
##     40        0.6483             nan     0.1000    0.0085
##     60        0.5155             nan     0.1000    0.0064
##     80        0.4203             nan     0.1000    0.0032
##    100        0.3486             nan     0.1000    0.0032
##    120        0.2973             nan     0.1000    0.0023
##    140        0.2558             nan     0.1000    0.0020
##    150        0.2375             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2453
##      2        1.4554             nan     0.1000    0.1509
##      3        1.3509             nan     0.1000    0.1218
##      4        1.2707             nan     0.1000    0.1057
##      5        1.1998             nan     0.1000    0.0934
##      6        1.1403             nan     0.1000    0.0791
##      7        1.0858             nan     0.1000    0.0591
##      8        1.0437             nan     0.1000    0.0677
##      9        0.9995             nan     0.1000    0.0483
##     10        0.9655             nan     0.1000    0.0465
##     20        0.7272             nan     0.1000    0.0209
##     40        0.4930             nan     0.1000    0.0129
##     60        0.3605             nan     0.1000    0.0065
##     80        0.2770             nan     0.1000    0.0046
##    100        0.2193             nan     0.1000    0.0025
##    120        0.1804             nan     0.1000    0.0024
##    140        0.1465             nan     0.1000    0.0016
##    150        0.1327             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1280
##      2        1.5235             nan     0.1000    0.0869
##      3        1.4644             nan     0.1000    0.0653
##      4        1.4220             nan     0.1000    0.0611
##      5        1.3817             nan     0.1000    0.0432
##      6        1.3512             nan     0.1000    0.0398
##      7        1.3247             nan     0.1000    0.0404
##      8        1.2989             nan     0.1000    0.0316
##      9        1.2771             nan     0.1000    0.0385
##     10        1.2499             nan     0.1000    0.0277
##     20        1.0964             nan     0.1000    0.0169
##     40        0.9179             nan     0.1000    0.0098
##     60        0.8027             nan     0.1000    0.0070
##     80        0.7200             nan     0.1000    0.0041
##    100        0.6530             nan     0.1000    0.0017
##    120        0.5970             nan     0.1000    0.0023
##    140        0.5505             nan     0.1000    0.0017
##    150        0.5302             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1851
##      2        1.4875             nan     0.1000    0.1390
##      3        1.4019             nan     0.1000    0.0990
##      4        1.3352             nan     0.1000    0.0849
##      5        1.2811             nan     0.1000    0.0706
##      6        1.2348             nan     0.1000    0.0667
##      7        1.1921             nan     0.1000    0.0457
##      8        1.1594             nan     0.1000    0.0433
##      9        1.1290             nan     0.1000    0.0495
##     10        1.0970             nan     0.1000    0.0407
##     20        0.8782             nan     0.1000    0.0188
##     40        0.6661             nan     0.1000    0.0088
##     60        0.5317             nan     0.1000    0.0049
##     80        0.4390             nan     0.1000    0.0096
##    100        0.3663             nan     0.1000    0.0034
##    120        0.3105             nan     0.1000    0.0027
##    140        0.2673             nan     0.1000    0.0012
##    150        0.2479             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2363
##      2        1.4591             nan     0.1000    0.1580
##      3        1.3578             nan     0.1000    0.1249
##      4        1.2753             nan     0.1000    0.1051
##      5        1.2082             nan     0.1000    0.0714
##      6        1.1573             nan     0.1000    0.0796
##      7        1.1052             nan     0.1000    0.0698
##      8        1.0572             nan     0.1000    0.0603
##      9        1.0144             nan     0.1000    0.0652
##     10        0.9714             nan     0.1000    0.0463
##     20        0.7384             nan     0.1000    0.0208
##     40        0.5000             nan     0.1000    0.0098
##     60        0.3717             nan     0.1000    0.0048
##     80        0.2914             nan     0.1000    0.0047
##    100        0.2319             nan     0.1000    0.0019
##    120        0.1868             nan     0.1000    0.0014
##    140        0.1521             nan     0.1000    0.0013
##    150        0.1376             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1368
##      2        1.5196             nan     0.1000    0.0876
##      3        1.4602             nan     0.1000    0.0685
##      4        1.4136             nan     0.1000    0.0536
##      5        1.3779             nan     0.1000    0.0386
##      6        1.3494             nan     0.1000    0.0422
##      7        1.3196             nan     0.1000    0.0440
##      8        1.2910             nan     0.1000    0.0305
##      9        1.2681             nan     0.1000    0.0312
##     10        1.2463             nan     0.1000    0.0357
##     20        1.0819             nan     0.1000    0.0129
##     40        0.9058             nan     0.1000    0.0069
##     60        0.7894             nan     0.1000    0.0048
##     80        0.7065             nan     0.1000    0.0048
##    100        0.6400             nan     0.1000    0.0021
##    120        0.5856             nan     0.1000    0.0032
##    140        0.5384             nan     0.1000    0.0021
##    150        0.5171             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1867
##      2        1.4836             nan     0.1000    0.1279
##      3        1.3968             nan     0.1000    0.0974
##      4        1.3324             nan     0.1000    0.0806
##      5        1.2783             nan     0.1000    0.0729
##      6        1.2307             nan     0.1000    0.0606
##      7        1.1893             nan     0.1000    0.0571
##      8        1.1504             nan     0.1000    0.0537
##      9        1.1145             nan     0.1000    0.0501
##     10        1.0828             nan     0.1000    0.0345
##     20        0.8756             nan     0.1000    0.0247
##     40        0.6529             nan     0.1000    0.0151
##     60        0.5173             nan     0.1000    0.0079
##     80        0.4233             nan     0.1000    0.0025
##    100        0.3548             nan     0.1000    0.0039
##    120        0.2993             nan     0.1000    0.0018
##    140        0.2561             nan     0.1000    0.0019
##    150        0.2386             nan     0.1000    0.0004
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2363
##      2        1.4570             nan     0.1000    0.1573
##      3        1.3528             nan     0.1000    0.1271
##      4        1.2718             nan     0.1000    0.1012
##      5        1.2065             nan     0.1000    0.0851
##      6        1.1492             nan     0.1000    0.0784
##      7        1.0957             nan     0.1000    0.0786
##      8        1.0457             nan     0.1000    0.0540
##      9        1.0086             nan     0.1000    0.0441
##     10        0.9770             nan     0.1000    0.0561
##     20        0.7284             nan     0.1000    0.0204
##     40        0.4912             nan     0.1000    0.0105
##     60        0.3583             nan     0.1000    0.0059
##     80        0.2800             nan     0.1000    0.0045
##    100        0.2216             nan     0.1000    0.0022
##    120        0.1772             nan     0.1000    0.0022
##    140        0.1450             nan     0.1000    0.0011
##    150        0.1315             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1346
##      2        1.5161             nan     0.1000    0.0976
##      3        1.4543             nan     0.1000    0.0699
##      4        1.4073             nan     0.1000    0.0559
##      5        1.3698             nan     0.1000    0.0522
##      6        1.3342             nan     0.1000    0.0415
##      7        1.3072             nan     0.1000    0.0419
##      8        1.2798             nan     0.1000    0.0316
##      9        1.2561             nan     0.1000    0.0357
##     10        1.2333             nan     0.1000    0.0280
##     20        1.0703             nan     0.1000    0.0176
##     40        0.8907             nan     0.1000    0.0104
##     60        0.7720             nan     0.1000    0.0056
##     80        0.6855             nan     0.1000    0.0036
##    100        0.6206             nan     0.1000    0.0034
##    120        0.5676             nan     0.1000    0.0023
##    140        0.5236             nan     0.1000    0.0017
##    150        0.5033             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2007
##      2        1.4773             nan     0.1000    0.1402
##      3        1.3873             nan     0.1000    0.1022
##      4        1.3203             nan     0.1000    0.0835
##      5        1.2660             nan     0.1000    0.0733
##      6        1.2179             nan     0.1000    0.0608
##      7        1.1752             nan     0.1000    0.0567
##      8        1.1370             nan     0.1000    0.0560
##      9        1.0999             nan     0.1000    0.0467
##     10        1.0682             nan     0.1000    0.0462
##     20        0.8539             nan     0.1000    0.0224
##     40        0.6341             nan     0.1000    0.0093
##     60        0.5003             nan     0.1000    0.0066
##     80        0.4141             nan     0.1000    0.0033
##    100        0.3506             nan     0.1000    0.0022
##    120        0.3002             nan     0.1000    0.0025
##    140        0.2578             nan     0.1000    0.0015
##    150        0.2388             nan     0.1000    0.0028
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2522
##      2        1.4506             nan     0.1000    0.1709
##      3        1.3405             nan     0.1000    0.1349
##      4        1.2550             nan     0.1000    0.0998
##      5        1.1896             nan     0.1000    0.0814
##      6        1.1363             nan     0.1000    0.0841
##      7        1.0816             nan     0.1000    0.0634
##      8        1.0385             nan     0.1000    0.0674
##      9        0.9931             nan     0.1000    0.0488
##     10        0.9586             nan     0.1000    0.0524
##     20        0.7073             nan     0.1000    0.0193
##     40        0.4787             nan     0.1000    0.0071
##     60        0.3560             nan     0.1000    0.0039
##     80        0.2734             nan     0.1000    0.0033
##    100        0.2182             nan     0.1000    0.0012
##    120        0.1780             nan     0.1000    0.0002
##    140        0.1482             nan     0.1000    0.0012
##    150        0.1343             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1387
##      2        1.5152             nan     0.1000    0.0958
##      3        1.4525             nan     0.1000    0.0726
##      4        1.4058             nan     0.1000    0.0574
##      5        1.3673             nan     0.1000    0.0539
##      6        1.3308             nan     0.1000    0.0430
##      7        1.3008             nan     0.1000    0.0352
##      8        1.2768             nan     0.1000    0.0393
##      9        1.2493             nan     0.1000    0.0342
##     10        1.2250             nan     0.1000    0.0277
##     20        1.0575             nan     0.1000    0.0144
##     40        0.8840             nan     0.1000    0.0073
##     60        0.7777             nan     0.1000    0.0086
##     80        0.6943             nan     0.1000    0.0022
##    100        0.6304             nan     0.1000    0.0024
##    120        0.5761             nan     0.1000    0.0018
##    140        0.5309             nan     0.1000    0.0017
##    150        0.5102             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2027
##      2        1.4792             nan     0.1000    0.1296
##      3        1.3919             nan     0.1000    0.1090
##      4        1.3204             nan     0.1000    0.0861
##      5        1.2619             nan     0.1000    0.0750
##      6        1.2141             nan     0.1000    0.0723
##      7        1.1694             nan     0.1000    0.0595
##      8        1.1283             nan     0.1000    0.0502
##      9        1.0953             nan     0.1000    0.0492
##     10        1.0630             nan     0.1000    0.0348
##     20        0.8592             nan     0.1000    0.0162
##     40        0.6471             nan     0.1000    0.0108
##     60        0.5139             nan     0.1000    0.0024
##     80        0.4269             nan     0.1000    0.0056
##    100        0.3557             nan     0.1000    0.0031
##    120        0.3054             nan     0.1000    0.0031
##    140        0.2641             nan     0.1000    0.0010
##    150        0.2468             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2523
##      2        1.4503             nan     0.1000    0.1712
##      3        1.3413             nan     0.1000    0.1259
##      4        1.2580             nan     0.1000    0.0953
##      5        1.1956             nan     0.1000    0.0923
##      6        1.1346             nan     0.1000    0.0747
##      7        1.0848             nan     0.1000    0.0711
##      8        1.0390             nan     0.1000    0.0638
##      9        0.9971             nan     0.1000    0.0520
##     10        0.9612             nan     0.1000    0.0584
##     20        0.7231             nan     0.1000    0.0235
##     40        0.4894             nan     0.1000    0.0058
##     60        0.3640             nan     0.1000    0.0075
##     80        0.2809             nan     0.1000    0.0045
##    100        0.2241             nan     0.1000    0.0021
##    120        0.1817             nan     0.1000    0.0017
##    140        0.1479             nan     0.1000    0.0012
##    150        0.1332             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2222
##      2        1.4557             nan     0.1000    0.1622
##      3        1.3522             nan     0.1000    0.1121
##      4        1.2788             nan     0.1000    0.0911
##      5        1.2175             nan     0.1000    0.0901
##      6        1.1562             nan     0.1000    0.0733
##      7        1.1088             nan     0.1000    0.0791
##      8        1.0583             nan     0.1000    0.0563
##      9        1.0189             nan     0.1000    0.0481
##     10        0.9860             nan     0.1000    0.0504
##     20        0.7414             nan     0.1000    0.0197
##     40        0.5063             nan     0.1000    0.0098
##     60        0.3831             nan     0.1000    0.0063
##     80        0.2990             nan     0.1000    0.0039
##    100        0.2408             nan     0.1000    0.0017
##    120        0.1975             nan     0.1000    0.0003
##    140        0.1656             nan     0.1000    0.0010
##    150        0.1518             nan     0.1000   -0.0001
```

```r
predgbm <- predict(model4,newdata =validation)
confusionMatrix(predgbm,validation$classe)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##              1              1              1              1              0 
## AccuracyPValue  McnemarPValue 
##              0              0
```

Results
-------------------------------------
The Random Forest Model has the largest accuracy=1; therefore it is chosen to predict the testing data set

```r
testing2=subset(testing,select=c(names))
Test_cases <- predict(model3,newdata =testing2)
data.frame(Test_cases)
```

```
##    Test_cases
## 1           B
## 2           A
## 3           B
## 4           A
## 5           A
## 6           E
## 7           D
## 8           B
## 9           A
## 10          A
## 11          B
## 12          C
## 13          B
## 14          A
## 15          E
## 16          E
## 17          A
## 18          B
## 19          B
## 20          B
```
