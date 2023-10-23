#Cargamos paquetes
library(tidymodels)
library(discrim)
library(corrr)
library(paletteer)
library(MASS)

library(tidyr)




# Ejercicio 1

#Extraer el train y test

## Preparamos los datos

### Entrenamiento y validacion
flattened_images <- matrix(mnist_data$train$images, nrow = dim(mnist_data$train$images)[1], ncol = 28*28)
df <- data.frame(label = mnist_data$train$labels)
df <- cbind(df, flattened_images)

### Testeo

flattened_images2 <- matrix(mnist_data$test$images, nrow = dim(mnist_data$test$images)[1], ncol = 28*28)
test_data <- data.frame(label = mnist_data$test$labels)
test_data <- cbind(test_data, flattened_images2)

## Transformar el train y test al estadístico que deseamos de solo 1, 3 y 5

df <- df  %>% filter(label == 1 | label == 3 | label == 5)

test_data <- test_data %>% filter(label == 1 | label == 3 | label == 5) # data testeo
test_data$label <- as.factor(test_data$label)

# Spliteamos datos de entrenamiento y validación para elegir el mejor modelo

set.seed(191654)
### Spliteamos la data para validación y entrenamiento
data_split <- rsample::initial_split(df, prop = .8, strata = "label")

train_data <- training(data_split) # data train
train_data$label <- as.factor(train_data$label)

val_data <- testing(data_split) # data validación
val_data$label <- as.factor(val_data$label)

# Preparación de datos para reducir dimensionalidad con PCA

# Preparamos motores de modelo


lda_spec <- discrim_linear() %>%
  set_mode("classification") %>%
  set_engine("MASS")

qda_spec <- discrim_quad() %>%
  set_mode("classification") %>%
  set_engine("MASS")

nb_spec <- naive_Bayes() %>% 
  set_mode("classification") %>% 
  set_engine("klaR") %>% 
  set_args(usekernel = FALSE)

lr_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Preparamos receta para primer pca

rec <- recipe(label ~ ., data = train_data) %>%
    #step_zv(all_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  step_pca(all_predictors(), num_comp=100)

# Realizamos PCA

prep_rec <- prep(rec, training = train_data)

train_data_pca <- bake(prep_rec, new_data = train_data)

# Get variance explained by each component
pca_variance <- tidy(prep_rec, number = 3, type = "variance")

filtered_data <- pca_variance[pca_variance$terms == "variance", ]

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


#Resultado para testear
resultado2 <- fit_modelos2(qda_spec, train_data)
  
resultados_test <- as.data.frame(list(resultado2))

# Agregar la columna "Modelo" con el valor "QDA"
resultados_test$Modelo <- "QDA"

# Reordenar las columnas para que "Modelo" sea la primera columna
resultados_test <- resultados_test[, c("Modelo", "Accuracy", "F_Measure", "Recall", "Precision", "MCC")]