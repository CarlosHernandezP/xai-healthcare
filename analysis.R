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

# 04 Create new categories for some variables ----
age_low <- c("01-04 years", "05-09 years", "10-14 years", "15-19 years", "20-24 years", "25-29 years", "30-34 years", "35-39 years", "40-44 years")
age_med <- c("45-49 years", "50-54 years", "55-59 years", "60-64 years")

primary_site_first <- c("C44.0-Skin of lip, NOS", "C44.1-Eyelid", "C44.2-External ear", "C44.3-Skin other/unspec parts of face", "C44.4-Skin of scalp and neck")
primary_site_second <- c("C44.5-Skin of trunk")

data_trans <- data_trans %>% 
  mutate(
    age_rec = case_when(age_cat %in% age_low ~ "age_low", age_cat %in% age_med ~ "age_intermediate", .default = "age_advanced"),
    race_rec = case_when(race == "White" ~ "white", .default = "other"),
    marital_status_rec = case_when(marital_status == "Married (including common law)"  ~ "married", .default = "other"),
    primary_site_rec = case_when(primary_site %in% primary_site_first ~ "first", primary_site %in% primary_site_second ~ "second", "third"),
  )


unique(data_trans$marital_status)
