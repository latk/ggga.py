# This R script is the counterpoint to ggga.examples.__main__.IraceStrategy.

library(irace)
library(jsonlite)

parameters.table <- '___PARAMS___'
parameters <- readParameters(text = parameters.table)

target.runner <- function(experiment, scenario) {
    seed = experiment$seed
    configuration <- experiment$configuration

    conn <- socketConnection(
        port = ___PORT___, blocking = TRUE, encoding = "UTF-8")
    writeLines("evaluation request", conn)
    writeLines(toJSON(list(seed = seed, params = configuration)), conn)
    flush(conn)
    # cat("<<< EVALUATION CLIENT >>> sent", paste(configuration), "\n")
    y <- fromJSON(readLines(con = conn))[["y"]]
    close(conn)

    return(list( cost = y ))
}

scenario <- list(
    targetRunner = target.runner,
    parallel = ___PARALLEL___,
    instances = 1:1,
    maxExperiments = ___N_SAMPLES___,
    seed = ___SEED___,
    digits = ___DIGITS___,
    firstTest = ___FIRST_TEST___,
    confidence = ___CONFIDENCE___,
    logFile = "")  # do not create a logfile
checkIraceScenario(scenario, parameters = parameters)

tuned.confs = irace(scenario = scenario, parameters = parameters)

