#Cargamos paquetes
library(tidymodels)
library(discrim)
library(corrr)
library(paletteer)
library(MASS)
library(dslabs)
library(tidyr)


# Cargamos bases de datos
mnist_data <- read_mnist()
data2 <- iris

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


