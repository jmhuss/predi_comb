###########################################################################
###########################################################################
###                                                                     ###
###  TITLE: mach_learn_eval                                             ###
###  CREATION DATE: 2022-06                                             ###
###  LAST CHANGE:  15.09.2022                                           ###
###                                                                     ###
###########################################################################
###########################################################################


randomForest.cla.eval <- function(formula, data = NULL, criterion = "OOB", max.mtry = NULL, na.action = NULL,
                                  seed = NULL, print = FALSE, path = NULL, cores = "all", ...){
  
  # This function applies a random Forest model for classification and
  # evaluates the contribution of each individual predictor as well as
  # their combination.
  # Predictors are added according to their contribution measured as
  # minimized error (specified by 'criterion').
  # The optimal 'mtry' (number of random predictors selected for each
  # decision node) is derived and applied for each combination, also by
  # minimizing the specified error.
  
  # NOTE: if the function crashes while a text file is written, the connection
  #       to the sink is still open and all further screen output will be
  #       written to the file.
  #       -> in case of a crash, execute 'sink()' to close the connection
  
  # Input: the same as the randomForest()-function, except:
  # criterion:   - type of error to be used as criterion to determine the
  #                optimum 'mtry' and contribution of predictors
  #                Options: - 'OBB' (Out-of-Bag error),
  #                         - each factor of the predicted variable,
  #                           referring to the class error of this factor
  #                         - 'mean' (average class error)
  #                         - 'max' (maximum class error)
  # max.mtry:    - if set, it determines the maximum 'mtry' value that is tried
  #                in the mtry-optimization process. If Null, max.try is set
  #                to twice the square-root of the numbers of predictors of the
  #                respective combination.
  # na.action:   - if 'omit', all samples with NAs are dropped. If 'impute',
  #                missing values in the predictors are approximated using
  #                nfImpute() with an iteration of 6. This does not apply for
  #                level 1 (only one predictor), since the impute function
  #                requires > 1 predictor - samples are omitted instead.
  #                Samples with NAs in the predicted variable are omitted
  #                as well since they can't be approximated by the impute
  #                function.
  #                If the ntree argument (or other randomForest()-arguments
  #                are passed via '...', they are also applied to rfImpute())
  # seed:        - integer used to set the seed. If NULL', random seeds are
  #                produced.
  # print:       - if TRUE the OBB error and confusion matrix with class
  #                errors is printed for each combination. If FALSE, only
  #                the selected criterion error is printed for each levels
  #                best combination
  # path:        - if NULL, all screen output will be printed to the console.
  #                If a path (without file name but ending with '/') is
  #                defined, screen output will instead be written to a
  #                text file:
  #                randForest_cla_eval_by<criterion>-error<datetime>.txt
  
  # Output is a list, containing:
  # - a list with the model results for each rund (predictor combination)
  #   in each level (1 to all available predictors)
  # - a data.frame with all prediction errors for each run
  # - the used seed (for reproducibility)
  # - the optimization criterion
  
  # install the random forest package if it is not installed yet
  if(!require(randomForest)){
    cat("install dependencies:")
    install.packages("randomForest")}
  library(randomForest)
  library(parallel)
  library(foreach)
  
  # throw an error if 'mtry' is given via '...' since it would interfer with the mtry-optimization
  if("mtry" %in% names(match.call()))
    stop("'mtry' must not be defined since the optimal mtry is calculated and applied ",
         "autimatically for each run, minimizing the specified error")
  
  # check if 'print' is a logical statement
  if(!is.logical(print))
    stop("'print' must be 'TRUE' or 'FALSE'")
  
  # assign number of cores for parralel computing
  if(cores == "all"){
    cores <- detectCores()
  }else if(is.numeric(cores)){
    if(cores > detectCores()){
      cores <- detectCores()
      warning(paste0("'cores' exceeds availables logical core count. ",cores," threads are used"))
    }
  }else{
    stop("'cores' must be 'all' or a number")
  }
  # start cluster for parallel computing
  cl <- parallel::makeCluster(cores)
  doParallel::registerDoParallel(cl)
  
  # prepare seeds
  if(is.null(seed)){
    # randomly select a seedn
    seed <- round(runif(1, 1,1e5))
  }else{
    if(round(seed) != seed){
      seed <- round(seed)
      warning("'seed' was converted to an integer")
    }
  }
  # set seed for replicability of the random component of the model
  set.seed(seed)
  
  
  ###=================================###
  ### prepare input and empty objects ###
  ###=================================###
  
  # let the prep.input() function check the formula and data input and prepare them
  data <- prep.input(formula, data)
  # the target variable as vector
  target <- data$target
  # check if the target variable is a factor
  if(!is.factor(target))
    stop("The predicted variable must be a factor. Else, use randomForest.reg.eval()")
  # all predictors as data.frame
  pred <- data$pred
  rm(data)
  
  # categories to be predicted (levels of 'target')
  fact <- as.character(unique(target[which(!is.na(target))]))
  
  # sequence along the total number of predictors
  pr <- 1:ncol(pred)
  
  # empty objects to dump all the model and error results
  res <- list()
  errors <- data.frame((matrix(nrow = sum(pr), ncol = 7+length(fact))))
  names(errors) <- c("level","pars","new_par","selected","OOB",rep(NA,length(fact)),"mean","max")
  errors$selected <- F
  # empty vector to drop the indetifyer of each runs best new predictor
  fixed <- c()
  
  # count total runs
  run <- 0
  
  if(!is.null(path)){
    # open connection to write screen output to text file instead
    cat("text file is written, no screen output appears\n")
    name <- paste0("randForest_cla_eval by ",criterion,"-error ",Sys.time(),".txt")
    # replace ":" in the fiel name since it not allowed in some systems
    name <- gsub(":","_",name)
    sink(paste0(path,name))
  }
  
  ###======================###
  ### main processing loop ###
  ###======================###
  
  # iterate over levels (no. of predictors)
  for(l in seq_along(pr)){
    
    # print current level
    cat("### level ",l,"/",length(pr),"\n", sep = "")
    
    # runs (combinations of predictors) for this level
    #  (all but those already preselected from previous iterations and
    #  hence included in each run)
    if(length(fixed) == 0){
      combs <- pr
    }else{
      combs <- pr[-fixed]
    }
    # empty list in list to dump all model results of the recent level
    res[[l]] <- list()
    
    # iterate over combinations in current level
    for(c in seq_along(combs)){
      run <- run+1
      
      # create label (string with combination of current predictors)
      if(l == 1){
        pars <- names(pred)[combs[c]]
      }else{
        pars <- paste(paste(names(pred)[fixed], collapse = "+"),names(pred)[combs[c]], sep = "+")
      }
      # subset data.frame with only the required predictors for this run
      sub_df <- data.frame(target, pred[c(fixed,combs[c])])
      
      # omit all columns where the predicted variable is NA
      sub_df <- sub_df[!is.na(sub_df$target),]
      
      
      # in no or non-recognized na.action found: ask user
      if(is.null(na.action)){
        na.action <- "none"
      }
      if(any(is.na(sub_df)) & !(na.action %in% c("omit","impute"))){
        library("tcltk")
        button <- tk_messageBox(title="NAs found!", type=c('yesnocancel'), default = "no", icon = "question",
                                message = "Shall NAs in the predictors be imputed?\nElse, they are omitted.")
        # button <- tk_select.list(choices = c("omit","impute"), preselect = "omit",
        #                          multiple = FALSE, title = "NAs detected")
        if(button == 'yes'){
          na.action <- "impute"
        }else if(button == "no"){
          na.action <- "omit"
        }else{
          # close sink
          if(!is.null(path))
            sink()
          stop("run cancelled...")
        }
      }
      
      # apply na.action, if desired and necessary
      if(any(is.na(sub_df)) & (na.action == "omit" | l == 1)){
        # discard samples with missing vaues amongst the predictors
        sub_df <- na.omit(sub_df)
      }else if(na.action == "impute" & any(is.na(sub_df))){
        # approximate missing values amongst the predictors
        #  (screen output/file output is surpressed. If output desired,
        #  rfImpute() has to be taken out of capture.output())
        capture.output(sub_df <- rfImpute(target ~ ., data = sub_df, iter = 6, ...), file = nullfile())
      }
      
      # in case of more than one predictor:
      if(l > 1){
        ### optimize mtry (number of random predictors selected for each descision node)
        # iterate over possible values for 'mtry'
        # (if set: only try 'mtry's up to 'max.mtry')
        if(is.null(max.mtry)){
          #seq_mtry <- 1:(ncol(sub_df)-1)
          seq_mtry <- 1:(floor(sqrt((ncol(sub_df)-1)))*2)
        }else{
          if(max.mtry >= (ncol(sub_df)-1)){
            seq_mtry <- 1:(ncol(sub_df)-1)
          }else{
            seq_mtry <- 1:max.mtry
          }
        }
        
        # create empty data.frame to store the mtry-errors
        mtry_error <- data.frame(matrix(nrow = length(seq_mtry), ncol = 3+length(fact)))
        
        ####################################
        ### sequential computing
        
        # for(m in seq_mtry){
        #   model <- randomForest(target ~ ., data = sub_df, mtry = m, ...)
        #   # store the errors
        #   mtry_err.rate <- model$err.rate[nrow(model$err.rate),]
        #   names(mtry_error) <- c(names(mtry_err.rate),"mean","max")
        #   mtry_error[m,1:length(mtry_err.rate)] <- mtry_err.rate
        #   mtry_error$mean[m] <- mean(mtry_err.rate[-1])
        #   mtry_error$max[m] <- max(mtry_err.rate[-1])
        # }
        
        ####################################
        ### parallel computing
        
        mtry_err_temp <- foreach::foreach(m = seq_mtry, .packages = "randomForest") %dopar%{
          model <- randomForest(target ~ ., data = sub_df, mtry = m, ...)
          # store the errors
          mtry_err.rate <- c(model$err.rate[nrow(model$err.rate),],
                             mean(model$err.rate[nrow(model$err.rate),-1]),
                             max(model$err.rate[nrow(model$err.rate),-1]))
        }
        # extract values from list
        for(m in seq_along(mtry_err_temp)){
          mtry_error[m,] <- mtry_err_temp[[m]]
        }
        names(mtry_error) <- c(names(mtry_err_temp[[1]])[1:(length(mtry_err_temp[[1]])-2)],"mean","max")

        ####################################
        
        # select 'mtry' with lowest error specified as 'criterion'
        mtry_opt <- which(mtry_error[,criterion] == min(mtry_error[,criterion]))[1]
      }else{
        # at the first level only one predictor is available
        mtry_opt <- 1
      }
      # (re)run with optimal mtry
      res[[l]][[c]] <- randomForest(target ~ ., data = sub_df, mtry = mtry_opt, ...)
      
      # store all error rates
      err.rate <- res[[l]][[c]]$err.rate[nrow(res[[l]][[c]]$err.rate),]
      # (in first run, assign correct labels to the class.errors)
      if(run == 1)
        names(errors)[6:(5+length(fact))] <- names(err.rate)[-1]
      errors$level[run] <- l
      errors$pars[run] <- pars
      errors$new_par[run] <- names(pred)[combs[c]]
      errors[run,5:(5+length(fact))] <- err.rate
      errors$mean[run] <- mean(err.rate[-1])
      errors$max[run] <- max(err.rate[-1])
      
      # print errors for each combination
      if(print){
        # print respective combination of predictors
        cat(pars,"\n")
        # print OOB error
        cat("OOB:",round(errors$OOB[run],4),"\n")
        # print confusion matrix with class errors
        print(res[[l]][[c]]$confusion)
        cat("-------------------------------------------\n")
      }
      
    }# end iterate over combinations
    
    # name the result list after the contained predictors
    if(l == 1){
      names(res[[l]]) <- names(pred)[combs]
    }else{
      names(res[[l]]) <- paste(paste(names(pred)[fixed], collapse = "+"),names(pred)[combs], sep = "+")
    }
    # check, if the selected error ('criterion') is a valid option
    if(l == 1)
      if(!(criterion %in% names(errors)[-c(1:4)]))
        stop("'criterion' must be one out of '",paste(names(errors)[-c(1:4)], collapse = "' '"),"'")
    
    # add parameter with lowest error to 'fixed' list for the next level
    # and set 'selected' in 'errors' to TRUE for the respective combination
    sel_level <- which(errors$level == l)
    sel_error <- errors[sel_level,criterion]
    opt <- which(sel_error == min(sel_error))[1]
    fixed <- c(fixed, combs[opt])
    errors$selected[sel_level[opt]] <- T
    
    # print optimal predictor combination and its performance
    cat("-> optimum:\n",paste(names(pred)[fixed], collapse = " + "), sep = "")
    cat("\n1 -",criterion,"error = ",round(1-sel_error[opt],4),"\n\n")
    if(print)
      cat("===========================================\n")
    
  }# end iterate over levels
  
  # close cluster for parallel computing
  parallel::stopCluster(cl); rm(cl)
  
  # close connection to sink .txt-file
  if(!is.null(path))
    sink()
  
  # make the levels in the error data.frame a factor
  errors$level <- as.factor(errors$level)
  
  return(list(model_results = res, errors = errors, seed = seed, criterion = criterion))
  
}# end randomForest.cla.eval()


#=============================================================
#=============================================================


randomForest.reg.eval <- function(formula, data = NULL, criterion = "rmse", max.mtry = NULL, na.action = NULL,
                                  seed = NULL, print = FALSE, path = NULL, cores = "all", ...){
  
  # This function applies a random Forest model for regression and
  # evaluates the contribution of each individual predictor as well as
  # their combination.
  # Predictors are added according to their contribution measured as
  # minimized error (specified by 'criterion').
  # The optimal 'mtry' (number of random predictors selected for each
  # decision node) is derived and applied for each combination, also by
  # minimizing the specified error.
  
  # NOTE: if the function crashes while a text file is written, the connection
  #       to the sink is still open and all further screen output will be
  #       written to the file.
  #       -> in case of a crash, execute 'sink()' to close the connection
  
  # Input: the same as the randomForest()-function, except:
  # criterion:   - type of error to be used as criterion to determine the
  #                optimum 'mtry' and contribution of predictors
  #                Options: - 'rmse' (root mean squared error),
  #                         - 'rsq' (pseudo R-squared: 1 - mse / Var(y))
  # max.mtry:    - if set, it determines the maximum 'mtry' value that is tried
  #                in the mtry-optimization process. If Null, max.try is set
  #                to the floor of the the numbers of predictors / 1.5 of the
  #                respective combination.
  # na.action:   - if 'omit', all samples with NAs are dropped. If 'impute',
  #                missing values in the predictors are approximated using
  #                nfImpute() with an iteration of 6. This does not apply for
  #                level 1 (only one predictor), since the impute function
  #                requires > 1 predictor - samples are omitted instead.
  #                Samples with NAs in the predicted variable are omitted
  #                as well since they can't be approximated by the impute
  #                function.
  #                If the ntree argument (or other randomForest()-arguments
  #                are passed via '...', they are also applied to rfImpute())
  # seed:        - integer used to set the seed. If NULL', random seeds are
  #                produced.
  # print:       - if TRUE the mse and rsq are printed for each
  #                combination. If FALSE, only the selected criterion error
  #                is printed for each levels best combination
  # path:        - if NULL, all screen output will be printed to the console.
  #                If a path (without file name but ending with '/') is
  #                defined, screen output will instead be written to a
  #                text file:
  #                randForest_reg_eval_by<criterion>-error<datetime>.txt
  
  # Output is a list, containing:
  # - a list with the model results for each rund (predictor combination)
  #   in each level (1 to all available predictors)
  # - a data.frame with all prediction errors for each run
  # - the used seed (for reproducibility)
  # - the optimization criterion
  
  # install the random forest package if it is not installed yet
  if(!require(randomForest)){
    cat("install dependencies:")
    install.packages("randomForest")}
  library(randomForest)
  library(parallel)
  library(foreach)
  
  if(!(criterion %in% c("rmse","rsq")))
    stop("'criterio' must be 'rmse' or 'rsq'")
  
  # throw an error if 'mtry' is given via '...' since it would interfer with the mtry-optimization
  if("mtry" %in% names(match.call()))
    stop("'mtry' must not be defined since the optimal mtry is calculated and applied ",
         "autimatically for each run, minimizing the specified error")
  
  # check if 'print' is a logical statement
  if(!is.logical(print))
    stop("print must be 'TRUE' or 'FALSE'")
  
  # assign number of cores for parralel computing
  if(cores == "all"){
    cores <- detectCores()
  }else if(is.numeric(cores)){
    if(cores > detectCores()){
      cores <- detectCores()
      warning(paste0("'cores' exceeds availables logical core count. ",cores," threads are used"))
    }
  }else{
    stop("'cores' must be 'all' or a number")
  }
  # start cluster for parallel computing
  cl <- parallel::makeCluster(cores)
  doParallel::registerDoParallel(cl)
  
  # prepare seeds
  if(is.null(seed)){
    # randomly select a seedn
    seed <- round(runif(1, 1,1e5))
  }else{
    if(round(seed) != seed){
      seed <- round(seed)
      warning("'seed' was converted to an integer")
    }
  }
  # set seed for replicability of the random component of the model
  set.seed(seed)
  
  
  ###=================================###
  ### prepare input and empty objects ###
  ###=================================###
  
  # let the prep.input() function check the formula and data input and prepare them
  data <- prep.input(formula, data)
  # the target variable as vector
  target <- data$target
  # check if the target variable is a factor
  if(!is.numeric(target))
    stop("The predicted variable must be numeric. Else, use randomForest.cla.eval()")
  # all predictors as data.frame
  pred <- data$pred
  rm(data)
  
  # sequence along the total number of predictors
  pr <- 1:ncol(pred)
  
  # empty objects to dump all the model and error results
  res <- list()
  errors <- data.frame((matrix(nrow = sum(pr), ncol = 6)))
  names(errors) <- c("level","pars","new_par","selected","rmse","rsq")
  errors$selected <- F
  # empty vector to drop the indetifyer of each runs best new predictor
  fixed <- c()
  
  # count total runs
  run <- 0
  
  if(!is.null(path)){
    # open connection to write screen output to text file instead
    cat("text file is written, no screen output appears\n")
    name <- paste0("randForest_reg_eval by ",criterion," ",Sys.time(),".txt")
    # replace ":" in the fiel name since it not allowed in some systems
    name <- gsub(":","_",name)
    sink(paste0(path,name))
  }
  
  ###======================###
  ### main processing loop ###
  ###======================###
  
  # iterate over levels (no. of predictors)
  for(l in seq_along(pr)){
    
    # print current level
    cat("### level ",l,"/",length(pr),"\n", sep = "")
    
    # runs (combinations of predictors) for this level
    #  (all but those already preselected from previous iterations and
    #  hence included in each run)
    if(length(fixed) == 0){
      combs <- pr
    }else{
      combs <- pr[-fixed]
    }
    # empty list in list to dump all model results of the recent level
    res[[l]] <- list()
    
    # iterate over combinations in current level
    for(c in seq_along(combs)){
      run <- run+1
      
      # create label (string with combination of current predictors)
      if(l == 1){
        pars <- names(pred)[combs[c]]
      }else{
        pars <- paste(paste(names(pred)[fixed], collapse = "+"),names(pred)[combs[c]], sep = "+")
      }
      # subset data.frame with only the required predictors for this run
      sub_df <- data.frame(target, pred[c(fixed,combs[c])])
      
      # omit all columns where the predicted variable is NA
      sub_df <- sub_df[!is.na(sub_df$target),]
      
      
      # in no or non-recognized na.action found: ask user
      if(is.null(na.action)){
        na.action <- "none"
      }
      if(any(is.na(sub_df)) & !(na.action %in% c("omit","impute"))){
        library("tcltk")
        button <- tk_messageBox(title="NAs found!", type=c('yesnocancel'), default = "no", icon = "question",
                                message = "Shall NAs in the predictors be imputed?\nElse, they are omitted.")
        # button <- tk_select.list(choices = c("omit","impute"), preselect = "omit",
        #                          multiple = FALSE, title = "NAs detected")
        if(button == 'yes'){
          na.action <- "impute"
        }else if(button == "no"){
          na.action <- "omit"
        }else{
          # close sink
          if(!is.null(path))
            sink()
          stop("run cancelled...")
        }
      }
      
      # apply na.action, if desired and necessary
      if(any(is.na(sub_df)) & (na.action == "omit" | l == 1)){
        # discard samples with missing vaues amongst the predictors
        sub_df <- na.omit(sub_df)
      }else if(na.action == "impute" & any(is.na(sub_df))){
        # approximate missing values amongst the predictors
        #  (screen output/file output is surpressed. If output desired,
        #  rmImpute() has to be taken out of capture.output())
        capture.output(sub_df <- rfImpute(target ~ ., data = sub_df, iter = 6, ...), file = nullfile())
      }
      
      # in case of more than one predictor:
      if(l > 1){
        ### optimize mtry (number of random predictors selected for each descision node)
        # (if set: only try 'mtry's up to 'max.mtry')
        if(is.null(max.mtry)){
          #seq_mtry <- 1:(ncol(sub_df)-1)
          seq_mtry <- 1:max(floor((ncol(sub_df)-1)/1.5),1)
        }else{
          if(max.mtry >= (ncol(sub_df)-1)){
            seq_mtry <- 1:(ncol(sub_df)-1)
          }else{
            seq_mtry <- 1:max.mtry
          }
        }
        # create empty data.frame to store the mtry-errors
        mtry_error <- data.frame(matrix(nrow = length(seq_mtry), ncol = 2))
        names(mtry_error) <- c("rmse","rsq")
        
        # iterate over possible values for 'mtry'
        
        ####################################
        ### sequential computing
        
        # for(m in seq_mtry){
        #   model <- randomForest(target ~ ., data = sub_df, mtry = m, ...)
        #   # store the errors
        #   mtry_error$rmse[m] <- sqrt(model$mse[length(model$mse)])
        #   # since rsq improves with accuracy, use 1-rsq to convert it
        #   #  to an error rate instead (-> variability NOT explained)
        #   mtry_error$rsq[m] <- 1-model$rsq[length(model$rsq)]
        # }
        
        ####################################
        ### parallel computing
        
        mtry_err_temp <- foreach::foreach(m = seq_mtry, .packages = "randomForest") %dopar%{
          model <- randomForest(target ~ ., data = sub_df, mtry = m, ...)
          # store the errors
          mtry_err.rate <- c(sqrt(model$mse[length(model$mse)]),
                             1-model$rsq[length(model$rsq)])
        }
        # extract values from list
        for(m in seq_along(mtry_err_temp)){
          mtry_error[m,] <- mtry_err_temp[[m]]
        }
        
        ####################################
        
        # select 'mtry' with lowest error specified as 'criterion'
        mtry_opt <- which(mtry_error[,criterion] == min(mtry_error[,criterion]))[1]
      }else{
        # at the first level only one predictor is available
        mtry_opt <- 1
      }
      # (re)run with optimal mtry
      res[[l]][[c]] <- randomForest(target ~ ., data = sub_df, mtry = mtry_opt, ...)
      
      # store errors
      errors$level[run] <- l
      errors$pars[run] <- pars
      errors$new_par[run] <- names(pred)[combs[c]]
      errors$rmse[run] <- sqrt(res[[l]][[c]]$mse[length(res[[l]][[c]]$mse)])
      errors$rsq[run] <- 1-res[[l]][[c]]$rsq[length(res[[l]][[c]]$rsq)]
      
      # print errors for each combination
      if(print){
        # print respective combination of predictors
        cat(pars,"\n")
        # print OOB error
        cat("RMSE:",round(errors$rmse[run],4),"\n")
        cat("rsq:",round(1-errors$rsq[run],4),"\n")
        cat("-------------------------------------------\n")
      }
      
    }# end iterate over combinations
    
    # name the result list after the contained predictors
    if(l == 1){
      names(res[[l]]) <- names(pred)[combs]
    }else{
      names(res[[l]]) <- paste(paste(names(pred)[fixed], collapse = "+"),names(pred)[combs], sep = "+")
    }

    # add parameter with lowest error to 'fixed' list for the next level
    # and set 'selected' in 'errors' to TRUE for the respective combination
    sel_level <- which(errors$level == l)
    sel_error <- errors[sel_level,criterion]
    opt <- which(sel_error == min(sel_error))[1]
    fixed <- c(fixed, combs[opt])
    errors$selected[sel_level[opt]] <- T
    
    # print optimal predictor combination and its performance
    cat("-> optimum:\n",paste(names(pred)[fixed], collapse = " + "), sep = "")
    if(criterion == "rmse"){
      cat("\nRMSE = ",round(sel_error[opt],4),"\n\n")
    }else if(criterion == "rse"){
      cat("\nrsq = ",round(1-sel_error[opt],4),"\n\n")
    }
    
    if(print)
      cat("===========================================\n")
    
  }# end iterate over levels
  
  # close cluster for parallel computing
  parallel::stopCluster(cl); rm(cl)
  
  # close connection to sink .txt-file
  if(!is.null(path))
    sink()
  
  # make the levels in the error data.frame a factor
  errors$level <- as.factor(errors$level)
  # reconvert 1-rsq to rsq for final output
  errors$rsq <- 1 - errors$rsq
  
  return(list(model_results = res, errors = errors, seed = seed, criterion = criterion))
  
}# end randomForest.reg.eval()


#=============================================================
#=============================================================


randomForest.cla.avg <- function(formula, data = NULL, n = 3, seeds = NULL, ...){
  
  # This function runs the function randomForest.cla.eval() n times with
  # random (if not preset) seeds.
  # It compares the optimization path of predictors between the runs
  # and averages all error rates for runs with the same path.
  
  # Input as in randomForest.cla.eval(). Additionally:
  # n:       - number of runs
  # seeds:   - vector of integers used to set the seed for each run.
  #            If 'NULL', random seeds are produced.
  
  # Output is a list, containing:
  # - the error rates as data.frames (numbered), same as the output
  #   of randomForest.cla.eval(). The model results are dropped.
  # - a list ('avg') with the average error rates of each occurring
  #   optimization path as data.frame in order of their number of
  #   ocurrence. Their name specifies this number of occurrence.
  #   NOTE: this list is only produced, if any optimization path
  #         occurres more than once!
  # - a table with the average confusion matrix for the final run
  #   within each run along n (all predictors included). The average
  #   includes all runs irrespective of their optimization path
  # - a list with each ocurring optimization path and the index of
  #   all runs showing this path (as vector)
  # - a vector with the seeds used for each run for reproducibility
  # - the applied optimization criterion
  
  # prepare seeds
  if(is.null(seeds)){
    # randomly select seeds for each run
    seeds <- round(runif(n, 1,1e5))
  }else{
    if(length(seeds) != n)
      stop("'seeds' must either be 'NULL' or of legth 'n'")
    if(any(round(seeds) != seeds)){
      seeds <- round(seeds)
      warning("'seeds' were converted to integers")
    }
  }
  
  # start progress bar
  pb <- txtProgressBar(min = 0, max = n, style = 3, width = 75, char = "=")
  
  # prepare empty list to store error data
  errors <- confus <- vector(mode = 'list',n)
  names(errors) <- names(confus) <- 1:n
  
  ###=============================###
  ### run optimization repeatedly ###
  ###=============================###
  
  # run model optimization n times
  for(r in 1:n){
    #cat("run ",r,"/",n,"\n=========\n\n", sep = "")
    # surpress screen output
    capture.output(model <- randomForest.cla.eval(formula = formula, data = data,
                                                  seed = seeds[r], ...), file = nullfile())
    # only for troubleshooting (without '...')
    #model <- randomForest.cla.eval(formula = formula, data = data)
    errors[[r]] <- model$errors
    confus[[r]] <- model$model_results[[length(model$model_results)]][[1]]$confusion
    # read the used optimization criterion
    if(r == 1)
      crit <- model$criterion
    # update progress bar
    setTxtProgressBar(pb, r)
  }; rm(model)
  # close progress bar
  close(pb); rm(pb)
  cat("\n")
  
  
  ###================================###
  ### seperate by optimization paths ###
  ###================================###
  
  # length of data.frames in errors (number of rows)
  le <- nrow(errors[[1]])
  paths <- vector(length = n)
  # iterate over errors of each run
  for(e in seq_along(errors)){
    paths[e] <- errors[[e]]$pars[le]
  }
  tab <- sort(table(paths), decreasing = T)
  # create empty list to store, which runs showed which path
  paths_which <- vector(mode = 'list',length(tab))
  names(paths_which) <- names(tab)
  
  # if less paths than n
  if(length(tab) < n){
    
    # inform, if all runs had the same path
    if(length(tab) == 1)
      cat("Only one path oaccures\n\n")
    
    # prepare empty list for averages
    errors$avg <- list()
    
    # iterate over occurring paths
    for(t in seq_along(tab)){
      # print paths
      if(t == 1){
        cat("Most frequent path (",tab[t],"/",n,"):\n",names(tab)[t],"\n\n", sep = "")
      }else{
        cat("Next (",tab[t],"/",n,"):\n",names(tab)[t],"\n\n", sep = "")
      }
      # all runs with current path
      use <- which(paths == names(tab)[t])
      # store which runs had the current path
      paths_which[[t]] <- use
      # copy one data.frame as template to write the averaged values to
      errors$avg[[t]] <- errors[[use[1]]]
      # only average if the current path occurres more than once
      if(length(use) > 1){
        
        ###======================###
        ### average eorror rates ###
        ###======================###
        
        # iterate over all columns in errors with numeric data
        for(c in 5:ncol(errors$avg[[t]])){
          # prepare empty data.frame to store current error parameter
          # of each run with the current path
          to_avg <- data.frame(matrix(nrow = le, ncol = length(use)))
          # iterate over each run with the current path
          for(u in seq_along(use)){
            # pick respective column of each run with the current path
            to_avg[,u] <- errors[[use[u]]][,c]
          }
          # average respective colum of each selected run
          errors$avg[[t]][,c] <- rowMeans(to_avg)
        }
      }
    }
    # name the averaging data.frames after the number of occurrence of
    # the respective optimization path
    names(errors$avg) <- paste0("x",tab)
  }else{
    cat("\nNo path of predictors occures more than once: nothing to average\n",
        "Only the raw errors of each run are returned\nOcurring paths are:\n\n", sep = "")
    cat(paste(paths, collapse = "\n"),"\n\n")
    # store which run had which path
    names(paths_which) <- paths
    for(p in seq_along(paths_which)){
      paths_which[[p]] <- p
    }
  }
  
  confus_avg <- confus[[1]]
  
  # iterate over all columns in confus
  for(c in 1:ncol(confus_avg)){
    # prepare empty data.frame to store current error parameter
    # of each run with the current path
    to_avg <- data.frame(matrix(nrow = nrow(confus_avg), ncol = n))
    # iterate over each run with the current path
    for(r in 1:n){
      # pick respective column of each run with the current path
      to_avg[,r] <- confus[[r]][,c]
    }
    # average respective colum of each selected run
    confus_avg[,c] <- rowMeans(to_avg)
  }
  # round class errors to 5 digits
  confus_avg[,ncol(confus_avg)] <- round(confus_avg[,ncol(confus_avg)],5)
  # print average confusion matrix for current path
  cat("Average confusion matrix:\n\n")
  print(confus_avg)
  cat("\n")
  
  # add average confusion matrices to output
  errors$confusion <- confus_avg
  # add each occurring path and by which of the n runs it was produced
  # to output
  errors$paths <- paths_which
  # add used seeds to output
  errors$seeds <- seeds
  # add optimization criterion to output
  errors$criterion <- crit
  
  return(errors)
  
}# end randomForest.cla.avg()


#=============================================================
#=============================================================


randomForest.reg.avg <- function(formula, data = NULL, n = 3, seeds = NULL, ...){
  
  # This function runs the function randomForest.reg.eval() n times with
  # random (if not preset) seeds.
  # It compares the optimization path of predictors between the runs
  # and averages all error rates for runs with the same path.
  
  # Input as in randomForest.reg.eval(). Additionally:
  # n:       - number of runs
  # seeds:   - vector of integers used to set the seed for each run.
  #            If 'NULL', random seeds are produced.
  
  # Output is a list, containing:
  # - the error rates as data.frames (numbered), same as the output
  #   of randomForest.reg.eval(). The model results are dropped.
  # - a list ('avg') with the average error rates of each occurring
  #   optimization path as data.frame in order of their number of
  #   ocurrence. Their name specifies this number of occurrence.
  #   NOTE: this list is only produced, if any optimization path
  #         occurres more than once!
  # - a table with the average confusion matrix for the final run
  #   within each run along n (all predictors included). The average
  #   includes all runs irrespective of their optimization path
  # - a list with each ocurring optimization path and the index of
  #   all runs showing this path (as vector)
  # - a vector with the seeds used for each run for reproducibility
  # - the applied optimization criterion
  
  # prepare seeds
  if(is.null(seeds)){
    # randomly select seeds for each run
    seeds <- round(runif(n, 1,1e5))
  }else{
    if(length(seeds) != n)
      stop("'seeds' must either be 'NULL' or of legth 'n'")
    if(any(round(seeds) != seeds)){
      seeds <- round(seeds)
      warning("'seeds' were converted to integers")
    }
  }
  
  # start progress bar
  pb <- txtProgressBar(min = 0, max = n, style = 3, width = 75, char = "=")
  
  # prepare empty list to store error data
  errors <- vector(mode = 'list',n)
  names(errors) <- 1:n
  
  ###=============================###
  ### run optimization repeatedly ###
  ###=============================###
  
  # run model optimization n times
  for(r in 1:n){
    #cat("run ",r,"/",n,"\n=========\n\n", sep = "")
    # surpress screen output
    capture.output(model <- randomForest.reg.eval(formula = formula, data = data,
                                                  seed = seeds[r], ...), file = nullfile())
    # only for troubleshooting (without '...')
    #model <- randomForest.cla.eval(formula = formula, data = data)
    errors[[r]] <- model$errors
    #confus[[r]] <- model$model_results[[length(model$model_results)]][[1]]$confusion
    # read the used optimization criterion
    if(r == 1)
      crit <- model$criterion
    # update progress bar
    setTxtProgressBar(pb, r)
  }; rm(model)
  # close progress bar
  close(pb); rm(pb)
  cat("\n")
  
  
  ###================================###
  ### seperate by optimization paths ###
  ###================================###
  
  # length of data.frames in errors (number of rows)
  le <- nrow(errors[[1]])
  paths <- vector(length = n)
  # iterate over errors of each run
  for(e in seq_along(errors)){
    paths[e] <- errors[[e]]$pars[le]
  }
  tab <- sort(table(paths), decreasing = T)
  # create empty list to store, which runs showed which path
  paths_which <- vector(mode = 'list',length(tab))
  names(paths_which) <- names(tab)
  
  # if less paths than n
  if(length(tab) < n){
    
    # inform, if all runs had the same path
    if(length(tab) == 1)
      cat("Only one path oaccures\n\n")
    
    # prepare empty list for averages
    errors$avg <- list()
    
    # iterate over occurring paths
    for(t in seq_along(tab)){
      # print paths
      if(t == 1){
        cat("Most frequent path (",tab[t],"/",n,"):\n",names(tab)[t],"\n\n", sep = "")
      }else{
        cat("Next (",tab[t],"/",n,"):\n",names(tab)[t],"\n\n", sep = "")
      }
      # all runs with current path
      use <- which(paths == names(tab)[t])
      # store which runs had the current path
      paths_which[[t]] <- use
      # copy one data.frame as template to write the averaged values to
      errors$avg[[t]] <- errors[[use[1]]]
      # only average if the current path occurres more than once
      if(length(use) > 1){
        
        ###======================###
        ### average eorror rates ###
        ###======================###
        
        # iterate over all columns in errors with numeric data
        for(c in 5:ncol(errors$avg[[t]])){
          # prepare empty data.frame to store current error parameter
          # of each run with the current path
          to_avg <- data.frame(matrix(nrow = le, ncol = length(use)))
          # iterate over each run with the current path
          for(u in seq_along(use)){
            # pick respective column of each run with the current path
            to_avg[,u] <- errors[[use[u]]][,c]
          }
          # average respective colum of each selected run
          errors$avg[[t]][,c] <- rowMeans(to_avg)
        }
      }
    }
    # name the averaging data.frames after the number of occurrence of
    # the respective optimization path
    names(errors$avg) <- paste0("x",tab)
  }else{
    cat("\nNo path of predictors occures more than once: nothing to average\n",
        "Only the raw errors of each run are returned\nOcurring paths are:\n\n", sep = "")
    cat(paste(paths, collapse = "\n"),"\n\n")
    # store which run had which path
    names(paths_which) <- paths
    for(p in seq_along(paths_which)){
      paths_which[[p]] <- p
    }
  }
  
  # confus_avg <- confus[[1]]
  # 
  # # iterate over all columns in confus
  # for(c in 1:ncol(confus_avg)){
  #   # prepare empty data.frame to store current error parameter
  #   # of each run with the current path
  #   to_avg <- data.frame(matrix(nrow = nrow(confus_avg), ncol = n))
  #   # iterate over each run with the current path
  #   for(r in 1:n){
  #     # pick respective column of each run with the current path
  #     to_avg[,r] <- confus[[r]][,c]
  #   }
  #   # average respective colum of each selected run
  #   confus_avg[,c] <- rowMeans(to_avg)
  # }
  # # round class errors to 5 digits
  # confus_avg[,ncol(confus_avg)] <- round(confus_avg[,ncol(confus_avg)],5)
  # # print average confusion matrix for current path
  # cat("Average confusion matrix:\n\n")
  # print(confus_avg)
  # cat("\n")
  # 
  # # add average confusion matrices to output
  # errors$confusion <- confus_avg
  
  # add each occurring path and by which of the n runs it was produced
  # to output
  errors$paths <- paths_which
  # add used seeds to output
  errors$seeds <- seeds
  # add optimization criterion to output
  errors$criterion <- crit
  
  return(errors)
  
}# end randomForest.cla.avg()


#=============================================================
#=============================================================


lda.eval <- function(formula, data = NULL, criterion = "max", print = FALSE, path = NULL, ...){
  
  # This function applies a Linear discriminant analysis (LDA) and evaluates
  # the contribution of each individual predictor as well as their combination.
  # Predictors are added according to their contribution measured as
  # minimized error (specified by 'criterion').
  
  # NOTE: if the function crashes while a text file is written, the connection
  #       to the sink is still open and all further screen output will be
  #       written to the file.
  #       -> in case of a crash, execute 'sink()' to close the connection
  
  
  # Input: the same as the lda()-function, except:
  # criterion:   - type of error to be used as criterion to determine the
  #                optimum 'mtry' and contribution of predictors
  #                Options: - each factor of the predicted variable,
  #                           referring to the class error of this factor
  #                         - 'mean' (average class error)
  #                         - 'max' (maximum class error)
  # print:       - if TRUE the OBB error and confusion matrix with class
  #                errors is printed for each combination. If FALSE, only
  #                the selected criterion error is printed for each levels
  #                best combination
  # path:        - if NULL, all screen output will be printed to the console.
  #                If a path (without file name but ending with '/') is
  #                defined, screen output will instead be written to a
  #                text file (LDA_eval_by<criterion>-error<datetime>.txt)
  
  # Output is a list, containing:
  # - a list with the model results for each rund (predictor combination)
  #   in each level (1 to all available predictors)
  # - a data.frame with all prediction errors for each run
  # - the optimization criterion
  
  # install the random forest package if it is not installed yet
  if(!require(MASS)){
    cat("install dependencies:")
    install.packages("MASS")}
  library(MASS)
  
  # check if 'print' is a logical statement
  if(!is.logical(print))
    stop("print must be 'TRUE' or 'FALSE'")
  
  
  ###=================================###
  ### prepare input and empty objects ###
  ###=================================###
  
  # let the prep.input() function check the formula and data input and prepare them
  data <- prep.input(formula, data)
  # the target variable as vector
  target <- data$target
  # check if the target variable is a factor
  if(!is.factor(target))
    stop("the predicted variable must be a factor")
  # all predictors as data.frame
  pred <- data$pred
  rm(data)
  
  # categories to be predicted
  fact <- as.character(unique(target[which(!is.na(target))]))
  
  # sequence along the total number of predictors
  pr <- 1:ncol(pred)
  
  # empty objects to dump all the model and error results
  res <- list()
  errors <- data.frame((matrix(nrow = sum(pr), ncol = 6+length(fact))))
  names(errors) <- c("level","pars","new_par","selected",rep(NA,length(fact)),"mean","max")
  errors$selected <- F
  # empty vector to drop the indetifyer of each runs best new predictor
  fixed <- c()
  
  # count total runs
  run <- 0
  
  if(!is.null(path)){
    # open connection to write screen output to text file instead
    cat("text file is written, no screen output appears\n")
    name <- paste0("LDA_eval by ",criterion,"-error ",Sys.time(),".txt")
    # replace ":" in the fiel name since it not allowed in some systems
    name <- gsub(":","_",name)
    sink(paste0(path,name))
  }
  
  
  ###======================###
  ### main processing loop ###
  ###======================###
  

  # iterate over levels (no. of predictors)
  for(l in seq_along(pr)){
    
    # print current level
    cat("### level ",l,"/",length(pr),"\n", sep = "")
    
    # runs (combinations of predictors) for this level
    # (all but those already preselected from previous iterations and
    # hence included in each run)
    if(length(fixed) == 0){
      combs <- pr
    }else{
      combs <- pr[-fixed]
    }
    # empty list in list to dump all model results of the recent level
    res[[l]] <- list()
    
    # iterate over combinations in current level
    for(c in seq_along(combs)){
      run <- run+1
      
      # create label (string with combination of current predictors)
      if(l == 1){
        pars <- names(pred)[combs[c]]
      }else{
        pars <- paste(paste(names(pred)[fixed], collapse = "+"),names(pred)[combs[c]], sep = "+")
      }
      # subset data.frame with only the required predictors for this run
      sub_df <- data.frame(target, pred[c(fixed,combs[c])])
      
      # run model
      res[[l]][[c]] <- lda(target ~ ., data = sub_df, CV = TRUE, ...)
      # build confusion matrix
      tab <- table(target, res[[l]][[c]]$class)
      # specific errors
      curr.error <- vector(length = nrow(tab))
      for(r in 1:nrow(tab)){
        curr.error[r] <- 1-(tab[r,r] / sum(tab[r,]))
      }
      
      # store the errors
      # (in first run, assign correct labels to the specific errors)
      if(run == 1)
        names(errors)[5:(4+length(fact))] <- colnames(tab)
      errors$level[run] <- l
      errors$pars[run] <- pars
      errors$new_par[run] <- names(pred)[combs[c]]
      errors[run,5:(4+length(fact))] <- curr.error
      errors$mean[run] <- mean(curr.error)
      errors$max[run] <- max(curr.error)
      
      # print errors for each combination
      if(print){
        # print respective combination of predictors
        cat(pars,"\n")
        # print confusion matrix and class errors
        # (first add class errors to confusion matrix table)
        tab <- cbind(tab,round(curr.error,4))
        colnames(tab)[length(colnames(tab))] <- "class.error"
        print(tab)
        cat("-------------------------------------------\n")
      }
      
    }# end iterate over combinations
    
    # name the result lists after the contained predictors
    if(l == 1){
      names(res[[l]]) <- names(pred)[combs]
    }else{
      names(res[[l]]) <- paste(paste(names(pred)[fixed], collapse = "+"),names(pred)[combs], sep = "+")
    }
    # check, if the selected error ('criterion') is a valid option
    if(l == 1)
      if(!(criterion %in% names(errors)[-c(1:4)]))
        stop("'criterion' must be one out of '",paste(names(errors)[-c(1:4)], collapse = "' '"),"'")
    
    # add parameter with lowest error to 'fixed' list for the next level
    # and set 'selected' in 'errors' to TRUE for the respective combination
    sel_level <- which(errors$level == l)
    sel_error <- errors[sel_level,criterion]
    opt <- which(sel_error == min(sel_error))[1]
    fixed <- c(fixed, combs[opt])
    errors$selected[sel_level[opt]] <- T
    
    # # add parameter with lowest error to 'fixed' list for the next level
    # sel_error <- errors[which(errors$level == l),criterion]
    # fixed <- c(fixed, combs[which(sel_error == min(sel_error))[1]])

    # print optimal predictor combination and its performance
    cat("-> optimum:\n",paste(names(pred)[fixed], collapse = " + "), sep = "")
    cat("\n1 -",criterion,"error = ",round(1-sel_error[opt],4),"\n\n")
    if(print)
      cat("===========================================\n")
  }# end iterate over levels
  
  # close connection to sink .txt-file
  if(!is.null(path))
    sink()
  
  # make the levels in the error data.frame a factor
  errors$level <- as.factor(errors$level)
  
  return(list(model_results = res, errors = errors, criterion = criterion))
  
}# end lda.eval()


#=============================================================
#=============================================================


prep.input <- function(formula, data){
  
  # This function is called by the *eval-functions. It takes the
  # formula and data input handed to the parent function, applies
  # some checks and prepares them in the format expected by the
  # parent function:
  # a list with the target variable as vector and all predictors in a
  # named data.frame.
  
  vars <- all.vars(formula)
  # if variables provided is data.frame:
  if(!is.null(data)){
    # check, if 'data' truely is da data.frame
    if(!is.data.frame(data))
      stop("'data' must be a data.frame")
    # assign target varibale
    if(vars[1] %in% names(data))
      target <- data[[vars[1]]]
    else
      stop("predicted variable not found in 'data'")
    
    # assign predictors
    if(vars[2] == "."){
      # use all predictors available in 'data'
      pred <- data[,which(names(data) != vars[1])]
    }else{
      # use specified predictors only
      match <- which(names(data) %in% vars[2:length(vars)])
      # ...if any mathces are found in data
      if(length(match) > 0){
        pred <- data[,match]
        # check, if all all selected predictors were found
        if(length(match) < (length(vars)-1))
          warning("'",paste(vars[which(!(vars[2:length(vars)] %in% names(pred)))+1], collapse = "' + '"),
                  "' not found in 'data'. Computation done without")
      }else
        stop("none of the predictors were found in 'data'")
    }
    # if variables defined from global environment
  }else{
    # assign target varibale
    if(exists(vars[1]))
      target <- eval(parse(text = vars[1]))
    else
      stop("predicted variable not found in global environment")
    
    # assign predictors
    # checl for existace of variables in the global environment
    match <- which(vars[2:length(vars)] %in% ls(envir = .GlobalEnv))
    # ...if any mathces are found in global environment
    if(length(match) > 0){
      pred <- data.frame(matrix(NA, nrow = length(target), ncol = length(match)))
      names(pred) <- vars[match+1] # +1 since 1 is the targetted varibale
      
      for(m in seq_along(match)){
        # get variable from global environment
        temp <- mget(vars[match[m]+1], envir = .GlobalEnv)[[1]]
        if(length(temp) != length(target))
          stop("length of variable ",vars[match[m]+1]," is different from length of the target variable")
        pred[,m] <- temp
      }; rm(temp, m)
      
      # check, if all all selected predictors were found
      if(length(match) < (length(vars)-1))
        warning("'",paste(vars[which(!(vars[2:length(vars)] %in% names(pred)))+1], collapse = "' + '"),
                "' not found in global environment. Computation done without")
    }else
      stop("none of the predictors were found in global environment")
  }
  
  return(list(target = target, pred = pred))
  
}# end prep.input()




