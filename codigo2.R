# Preparamos fit y workflows

set.seed(191654)

fit_modelos <- function(modelo_spec, data){
  workflow <- workflow() %>%
    add_recipe(rec) %>%
    add_model(modelo_spec)
  
  fit <- fit(workflow, data = train_data)
  
  f2 <- augment(fit, new_data = val_data) %>%
    accuracy(truth = label, estimate = .pred_class)
  
  f3 <- augment(fit, new_data = val_data) %>%
    f_meas(truth = label, estimate = .pred_class)
  
  f4 <- augment(fit, new_data = val_data) %>%
    recall(truth = label, estimate = .pred_class)
  
  f5 <- augment(fit, new_data = val_data) %>%
    precision(truth = label, estimate = .pred_class)
  
  f6 <- augment(fit, new_data = val_data) %>%
    mcc(truth = label, estimate = .pred_class)
  
  return(list(Accuracy = f2$.estimate, 
              F_Measure = f3$.estimate, 
              Recall = f4$.estimate, 
              Precision = f5$.estimate, 
              MCC = f6$.estimate))
              }

fit_modelos2 <- function(modelo_spec, data){
  workflow <- workflow() %>%
    add_recipe(rec) %>%
    add_model(modelo_spec)
  
  fit <- fit(workflow, data = train_data)
  
  f2 <- augment(fit, new_data = test_data) %>%
    accuracy(truth = label, estimate = .pred_class)
  
  f3 <- augment(fit, new_data = test_data) %>%
    f_meas(truth = label, estimate = .pred_class)
  
  f4 <- augment(fit, new_data = test_data) %>%
    recall(truth = label, estimate = .pred_class)
  
  f5 <- augment(fit, new_data = test_data) %>%
    precision(truth = label, estimate = .pred_class)
  
  f6 <- augment(fit, new_data = test_data) %>%
    mcc(truth = label, estimate = .pred_class)
  
  return(list(Accuracy = f2$.estimate, 
              F_Measure = f3$.estimate, 
              Recall = f4$.estimate, 
              Precision = f5$.estimate, 
              MCC = f6$.estimate))
              }

#arreglamos receta a 50 componentes

rec <- recipe(label ~ ., data = train_data) %>%
    #step_zv(all_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  step_pca(all_predictors(), num_comp=50)

# Asumiendo que tienes las especificaciones de los modelos en la siguiente lista:
lista_modelos <- list(QDA = qda_spec, LDA = lda_spec, LR = lr_spec, NB = nb_spec)

# Crear un dataframe vacío
resultados <- tibble(
  Modelo = character(),
  Accuracy = numeric(),
  F_Measure = numeric(),
  Recall = numeric(),
  Precision = numeric(),
  MCC = numeric()
)

# Bucle para entrenar cada modelo y guardar los resultados en el dataframe
for (nombre in names(lista_modelos)) {
  resultado <- fit_modelos(lista_modelos[[nombre]], data)
  
  resultados <- resultados %>%
    add_row(
      Modelo = nombre,
      Accuracy = resultado$Accuracy,
      F_Measure = resultado$F_Measure,
      Recall = resultado$Recall,
      Precision = resultado$Precision,
      MCC = resultado$MCC
    )%>%
    mutate(Promedio = (Accuracy + F_Measure + Recall + Precision + MCC) / 5)
}
