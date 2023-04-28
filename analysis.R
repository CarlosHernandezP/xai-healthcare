library(tidyverse)

data_folder <- "."
file_name <- "xai_train_data.csv"
file_foder <- file.path(data_folder, file_name)

#

# 01 read data ----
data <- readr::read_csv(file_foder)
new_cols <- c("age_cat", "age_yearly", "cs_tumor_size", "cs_extension", 
              "rx_summ_surg_prim_site", "gender", "race", "marital_status",
              "primary_site", "derived_ajcc_t", "derived_ajcc_n", "derived_ajcc_m",
              "summary_stage", "reason_no_cancer_directed_surgery", "radiation_recode",
              "chemotherapy_recode", "rx_summ_scope_reg_ln_sur", "rad_seq",
              "income", "vital_status_recode", "survival_months", "icd_O_3_hist_behav",
              "cod_to_site_recode_"
              )

data_cols <- data
colnames(data_cols) <- new_cols
View(data)
View(data_cols)
# 02 Data manipulation ----
# Create target variables
data_trans <- data_cols %>% 
  mutate(
    event_indicator = case_when(vital_status_recode == "Alive" ~ 0, .default = 1)
  )

data_trans %>% group_by(event_indicator, vital_status_recode) %>% count()

# 03 Analysis for categorical variables ----
data_cat <- data_trans %>% select_if(is.character)
data_cat$event_indicator <- data_trans$event_indicator
variables <- setdiff(colnames(data_cat), c("event_indicator"))

for (var in variables) {
  results <- data_cat %>%
    group_by(across(all_of(var))) %>% 
    summarise(
      total = n(), 
      event = sum(event_indicator), 
      rate_event = 100 * sum(event_indicator)/n()
    ) %>% 
    ungroup() %>% 
    mutate(rate_population = 100 * total/sum(total))
    
  print(results)
  print("-----------")
}


