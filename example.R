###########################################################################
###########################################################################
###                                                                     ###
###  TITLE: example                                                     ###
###  CREATION DATE: 2022-06                                             ###
###  LAST CHANGE:  15.09.2022                                           ###
###                                                                     ###
###########################################################################
###########################################################################

# preamble
{
  rm(list=ls()); gc()
  Sys.setenv("TZ"="UTC", LANG = "en")
  # Set working directory to file location (requires RStudio)
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  # Get and set the script name to label plots (requires RStudio)
  script <- paste0("-",substr(basename(rstudioapi::getActiveDocumentContext()$path),
                              1,nchar(basename(rstudioapi::getActiveDocumentContext()$path))-2))
  
  # create directory for output (logs and plots)
  dir.create("./output/")
  
  # load functions (assumes 'mach_learn_eval.R' directly in the working directory)
  source("./mach_learn_eval.R")
}


###############################
###  get a prepare dataset  ###
###############################

# The data preparation is taken from StatQuest (random_forest_demo):
# https://github.com/StatQuest/random_forest_demo/blob/master/random_forest_demo.R

{
  data <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header=FALSE)
  
  colnames(data) <- c(
    "age",
    "sex",# 0 = female, 1 = male
    "cp", # chest pain 
    # 1 = typical angina, 
    # 2 = atypical angina, 
    # 3 = non-anginal pain, 
    # 4 = asymptomatic
    "trestbps", # resting blood pressure (in mm Hg)
    "chol", # serum cholestoral in mg/dl
    "fbs",  # fasting blood sugar if less than 120 mg/dl, 1 = TRUE, 0 = FALSE
    "restecg", # resting electrocardiographic results
    # 1 = normal
    # 2 = having ST-T wave abnormality
    # 3 = showing probable or definite left ventricular hypertrophy
    "thalach", # maximum heart rate achieved
    "exang",   # exercise induced angina, 1 = yes, 0 = no
    "oldpeak", # ST depression induced by exercise relative to rest
    "slope", # the slope of the peak exercise ST segment 
    # 1 = upsloping 
    # 2 = flat 
    # 3 = downsloping 
    "ca", # number of major vessels (0-3) colored by fluoroscopy
    "thal", # this is short of thalium heart scan
    # 3 = normal (no cold spots)
    # 6 = fixed defect (cold spots during rest and exercise)
    # 7 = reversible defect (when cold spots only appear during exercise)
    "hd" # (the predicted attribute) - diagnosis of heart disease 
    # 0 if less than or equal to 50% diameter narrowing
    # 1 if greater than 50% diameter narrowing
  )
  
  ## First, replace "?"s with NAs.
  data[data == "?"] <- NA
  
  ## Now add factors for variables that are factors and clean up the factors
  ## that had missing data...
  data[data$sex == 0,]$sex <- "F"
  data[data$sex == 1,]$sex <- "M"
  data$sex <- as.factor(data$sex)
  
  data$cp <- as.factor(data$cp)
  data$fbs <- as.factor(data$fbs)
  data$restecg <- as.factor(data$restecg)
  data$exang <- as.factor(data$exang)
  data$slope <- as.factor(data$slope)
  
  data$ca <- as.integer(data$ca) # since this column had "?"s in it (which
  # we have since converted to NAs) R thinks that
  # the levels for the factor are strings, but
  # we know they are integers, so we'll first
  # convert the strings to integiers...
  data$ca <- as.factor(data$ca)  # ...then convert the integers to factor levels
  
  data$thal <- as.integer(data$thal) # "thal" also had "?"s in it.
  data$thal <- as.factor(data$thal)
  
  ## This next line replaces 0 and 1 with "Healthy" and "Unhealthy"
  data$hd <- ifelse(test=data$hd == 0, yes="Healthy", no="Unhealthy")
  data$hd <- as.factor(data$hd) # Now convert to a factor
}


#=====================#
#==                 ==#
#==  random Forest  ==#
#==                 ==#
#=====================#


### evaluate the best combination of parameters
###============================================

### Example 1 ###

randF <- randomForest.cla.eval(formula = hd ~ .,   # formula
                               data = data,        # get predictors and predicted variable from this data.frame
                               criterion = "max",  # optimize in a way that the largest class error is reduced
                               print = T,          # print detailed information about each run
                               path = "./output/") # write information to file in this path (and not to the console)
# no 'na.action' is defined but the data contains NAs -> user is
#  asked for action
# a random seed is produced


### Example 2 ###

hd <- data$hd
age <- data$age
cp <- data$cp
oldpeak <- data$oldpeak

randF <- randomForest.cla.eval(formula = hd ~ age + cp + oldpeak, # formula directly pointing to the variables
                               criterion = "Unhealthy",  # optimize in a way that the error for 'Unhealthy' is reduced
                               na.action = "impute",     # approximate values for NAs in the predicting variables
                               ntree = 300,              # make only 300 trees (default = 500)
                               seed = 42)                # use this specific seed
# print only basic information (optimal result for each level)
# write information the console (not to a file)
rm(hd,age,cp,oldpeak)


# If an error ocurs while output is directed to a file, the sink
# is not closed an all further console output (ecxept for errors)
# is written to that file. In this case, close the sink manually:
sink()


### repeat and average
###===================

randF_avg <- randomForest.cla.avg(formula = hd ~ .,
                                  data = data[,c(1:5,ncol(data))], # only use first 5 predictors for shorter run-time
                                  n = 5,                 # no. of times randomForest.cla.eval() shall be run
                                  criterion = "OOB",     # optimization for lowest Out-Of-Bag error
                                  na.action = "omit")    # samples with NAs are omited

# View mean errors for most frequet path
View(randF_avg$avg[[1]])
# get seeds (e.g. to rerun with altered settings but same
# random component)
seeds <- randF_avg$seeds



#===========#
#==       ==#
#==  LDA  ==#
#==       ==#
#===========#

# LDA can only handle continuous predictors
# > make subset of all numeric columns and the
#   response variable

data_sub <- data[,c(which(sapply(data, class) == "numeric"),
                    ncol(data))]

lda <- lda.eval(formula = hd ~ .,
                data = data_sub,
                criterion = "max",
                print = T,
                path = "./output/")
sink()




#=========================================#
#==                                     ==#
#==  plot driver performance automated  ==#
#==                                     ==#
#=========================================#

{
  # object name (output of randomForest.eval(), lda.eval() or summary.randForest())
  model <- "randF"
  #model <- "randF_avg"
  #model <- "lda"
  
  # optimization path, if 'model' is summary. 1 = most frequent path ...
  p <- 1
  
  ### file prefix and plot title
  prefix <- model
  
  ### file suffix (e.g. script name)
  suffix <- script
  
  ### path to save plot
  path <- "./output/"
  
  # extract data
  {
    if("errors" %in% names(eval(parse(text = model)))){
      errors <- eval(parse(text = model))$errors
      crit <- eval(parse(text = model))$criterion
    }else if("avg" %in% names(eval(parse(text = model)))){
      errors <- eval(parse(text = model))$avg[[p]]
      crit <- eval(parse(text = model))$criterion
    }else{
      stop("no 'errors' data found")
    }
  }
  
  ### plot only optimum of each level
  {
    ### colors to use
    col <- seq(1:10)
    
    ### subset for only the optimum combinations of each level
    sub <- errors[which(errors$selected),]
    
    ### x-axis labels (combination of predictors)
    labels <- sub$new_par
    
    ylim <- c(min(1-sub[,-(1:4)])-.05,1)
    lwd <- 1.5
    cex <- 2
    
    png(paste0(path,prefix,"_optimized_predictors_",crit,"",suffix,".png"), width = 22, height = 18, res = 400, units = "cm")
    par(mar = c(5,4,1,1.5))
    
    plot(NULL, NULL, xlim = c(1,nrow(sub)+.5), ylim = ylim, las = 1, xaxt = "n", xlab = "",
         ylab = "accuracy (-)", main = paste0(prefix," - optimization crit: ",crit))
    mtext(1, 4, text = "predictors")
    ### print x-axis
    {
      # ticks
      axis(1, at = 0:nrow(sub), labels = F)
      # vertical labels
      #axis(1, at = seq(0,nrow(sub)), labels = labels, tick = F, las = 2)
      # slightly rotated labels
      #text(seq(0,nrow(sub)), ylim[1]-range(ylim)/20, labels = labels, srt = 25, xpd = T)
      # horizontal labels with alternating line
      axis(1, at = seq(1,nrow(sub),2), labels = labels[seq(1,nrow(sub),2)], tick = F)
      axis(1, at = seq(2,nrow(sub),2), labels = labels[seq(2,nrow(sub),2)], tick = F, line = 1.2)
    }
    abline(v = seq(1,nrow(sub)), col = 8, lwd = .8)
    abline(h = axis(2, labels = F, tick = F), col = 8, lwd = .8)
    
    # columns with errors to be plotted
    if("OOB" %in% names(sub)){
      err_col <- 5:(ncol(sub)-2)
    }else{
      err_col <- 5:(ncol(sub)-1)
    }
    
    for(e in seq_along(err_col)){
      lines(x = 1:nrow(sub), y = 1-sub[,err_col[e]], type = "o", col = col[e], lwd = lwd, pch = 20, cex = cex)
    }
    legend("bottomright", legend = names(sub)[err_col],
           lwd = lwd, pch = 20, pt.cex = cex, col = 1:length(err_col),
           box.lty = 0, bg = adjustcolor("white", .75))
    box()
    dev.off()
    # clean up
    rm(col,sub,labels,ylim,lwd,cex,err_col,e)
  }
  
  ### plot all runs
  {
    ### colors to use
    col <- seq(1:10)
    
    ylim <- range(1-errors[,-(1:4)])
    #ylim <- c(.75,1)
    lwd <- 1.5
    cex <- 1.5
    
    x <- as.numeric(unique(errors$level))
    opt <- errors$pars[which(errors$selected)]
    
    ### x-axis labels (combination of predictors)
    labels <- c("-",opt[-length(opt)])
    
    png(paste0(path,prefix,"_all_predictors_",crit,"",suffix,".png"), width = 22, height = 18, res = 400, units = "cm")
    par(mar = c(5,4,1,1.5))
    
    plot(NULL, NULL, xlim = c(1,max(x)+.5), ylim = ylim, las = 1, xaxt = "n", xlab = "",
         ylab = "accuracy (-)", main = paste0(prefix," - optimization crit: ",crit))
    mtext(1, 4, text = "predictors")
    ### print x-axis
    {
      # ticks
      axis(1, at = seq_along(x), labels = F)
      # vertical labels
      #axis(1, at = seq(0,nrow(sub)), labels = labels, tick = F, las = 2)
      # slightly rotated labels
      #text(seq(0,nrow(sub)), ylim[1]-range(ylim)/20, labels = labels, srt = 25, xpd = T)
      # horizontal labels with alternating line
      axis(1, at = seq(1,max(x),2), labels = labels[seq(1,max(x),2)], tick = F)
      axis(1, at = seq(2,max(x),2), labels = labels[seq(2,max(x),2)], tick = F, line = 1.2)
    }
    abline(v = x, col = 8, lwd = .8)
    abline(h = axis(2, labels = F, tick = F), col = 8, lwd = .8)
    
    # columns with errors to be plotted
    if("OOB" %in% names(errors)){
      err_col <- 5:(ncol(errors)-2)
    }else{
      err_col <- 5:(ncol(errors)-1)
    }
    
    new_pars <- unique(errors$new_par)
    
    # iterate over levels
    for(l in 1:max(x)){
      if(l > 1){
        # omptimal combination of the previous level
        opt_prev <- which(errors$level == l-1 & errors$selected)
        origin <- errors[opt_prev,err_col]
      }
      sub <- errors[which(errors$level == l),]
      for(p in 1:ncol(sub)){
        for(e in seq_along(err_col)){
          if(l > 1)
            lines(x = c(l-1,l), y = c(1-origin[e], 1-sub[p,err_col[e]]), lwd = lwd, col = col[e])
          points(x = l, y = 1-sub[p,err_col[e]], cex = cex, pch = which(new_pars == sub$new_par[p]), col = col[e])
        }
      }
    }
    
    legend("bottomright", legend = c(new_pars, names(errors)[err_col]), pch = c(seq_along(new_pars),rep(NA,length(err_col))),
           lwd = c(rep(NA, length(new_pars)),rep(lwd, length(err_col))), col = c(rep("grey30",length(new_pars)), col[seq_along(err_col)]),
           box.lty = 0, bg = adjustcolor("white", .75))
    box()
    dev.off()
    # clean up
    rm(col,ylim,lwd,cex,x,opt,labels,err_col,new_pars,opt_prev,origin,sub,l,p,e)
  }
}




