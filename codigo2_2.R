# Preparamos fit y workflows

library(cluster)

#Iris DB
data <- as.matrix(iris[,1:4])
dist_mat <- dist(data, method = 'euclidean')

hclust_average <- hclust(dist_mat, method='average')
hclust_complete <- hclust(dist_mat, method= 'complete')
kmean_model <- kmeans(data, centers = 3)


library(caret)

cut_average <- cutree(hclust_average, k=3)
cut_complete <- cutree(hclust_complete, k=3)

#CHECANDO

true_labels <- iris$Species



accuracy_cut_av <- (sum(cut_average[1:50]==1)+sum(cut_average[51:100]==2)+ sum(cut_average[101:150]==3))/150



accuracy_cut_cp <- (sum(cut_complete[1:50]==1)+sum(cut_complete[51:100]==3)+ sum(cut_complete[101:150]==2))/150



accuracy_kmeans <- (sum(kmean_model$cluster[1:50]==3)+sum(kmean_model$cluster[51:100]==2)+ sum(kmean_model$cluster[101:150]==1))/150

# Datos para la tabla
methods <- c("Average Linkage", "Complete Linkage", "K-Means Clustering")
accuracies <- c(accuracy_cut_av, accuracy_cut_cp, accuracy_kmeans)

# Crear dataframe para la tabla
results_df <- data.frame(Method = methods, Accuracy = accuracies)
