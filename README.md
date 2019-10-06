# Clustering-De-Palabras

### Trabajo práctico
### Text mining, FaMAF, UNC

En el presente trabajo se hace clustering de palabras con un fragmento de la wikipedia en español.
Se aprovecha a comparar dos procedimientos seguidos:

### Procedimiento 1 - ClustersWiki.ipynb:

* Se carga el texto y se analiza con nlp() de Spacy.
* Se separan en oraciones y se limpian dejando solo los tokens que correspondan a palabras, dígitos, o el caracter '.'
* Se eliminan las oraciones con menos de 5 tokens.
* Se obtiene un dataframe con token, lema_, pos_, dep_ (en éste análisis solo se usan los lemas)
* Los __lemas__ se pasan a minúscula, y los dígitos se reemplazan por la palabra 'dígito'.
* Se cuenta la frecuencia de los lemas.
* Se __eliminan los lemas con una frecuencia menor a 50 y también se eliminan las stopwords del español__. Éstos no apareceran ni en las filas de la matriz, ni en las columnas como features.
* Se arma la __matriz de coocurrencias de palabras con una ventana de tamaño 2.__
* Al usar solo las frecuencias se usa __PPMI para normalizar__ la matriz.
* Se usa __TruncatedSVD__ de sklearn para reducir el tamaño de la matriz, conservando el 81% de la varianza.
* Se usa __t-sne con distancia euclidea para visualizar__ en dos dimensiones los agrupamientos.
* Se hace clustering con __kmeans de sklearn__ sobre la matriz reducida. Se vale de la __distancia euclidea__ para encontrar los clusters, no hay problema debido a que antes se normalizó la matriz.
  * Se itera tres veces a 10 , 50 y 100 clusters
  
### Procedimiento 2 - wikitriplas.py:

* Se carga el texto y se analiza con nlp() de Spacy.
* Se separan en oraciones y se limpian dejando solo los tokens que correspondan a palabras, dígitos, o el caracter '.'
* Se eliminan las oraciones con menos de 5 tokens.
* Se obtiene un dataframe con token, lema_, pos_, dep_
* Los __lemas__ se pasan a minúscula, y los dígitos se reemplazan por la palabra 'dígito'.
* Se cuenta la frecuencia de los lemas.
* Se __eliminan los lemas con una frecuencia menor a 50 y también se eliminan las stopwords del español__. Éstos no apareceran ni en las filas de la matriz, ni en las columnas como features.
* Se arma la __matriz de coocurrencias de palabras con una ventana de tamaño 2.__
* A la matriz de coocurrencias se le agregan los features de las __etiquetas pos__
* Se agregan los features de las __triplas de dependencia .dep .HeadDep__
* Se usa __TruncatedSVD__ de sklearn para reducir el tamaño de la matriz, conservando el 99% de la varianza.
* Se usa __t-sne con distancia coseno__ (ya que no se normalizó la matriz) para visualizar en dos dimensiones los agrupamientos.
* Se hace clustering con __kmeans de NLTK__ sobre la matriz reducida. Se usa la __distancia coseno__ para encontrar los clusters.
  * Se itera tres veces a 10 , 50 y 100 clusters
  
  
### Conclusiones: 

Los resultados más interesantes se pueden ver con 50 o 100 clusters. 

Al final de la notebook 'ClustersWiki.ipynb' que usa la matriz de frecuencias de palabras se pueden ver las palabras que caen en los 100 clusters. Si bien se consigue que palabras similares en cuanto a su funcionalidad (como por ejemplo, los días de la semana, las que describen números, los nombres de lugares, las profesiones) caigan en los mismos clusters, también se obtienen clusters que agrupan palabras relacionadas entre sí desde un aspecto más semántico y de tema del que se escribe (por ejemplo, palabras en inglés que escaparon en el preprocesado; otro con palabras que estarían en la temática 'guerra', palabras de geografía, de política, de música, palabras referentes a cargos de la realeza, palabras referentes al gobierno, palabras referentes al sistema educativo, al cine, a divisiones de la población, al deporte)
https://github.com/florealonso/Words_Clustering/blob/master/coocurrencias%20cluster/ClustersWiki.ipynb

En la notebook 'wikitriplas_n.ipynb' con los 100 clusters se puede observar que el uso de triplas de dependencia y morfología de las palabras hace que los clusters tengan más tendencia a contener palabras con la misma morfología, es decir, un cluster con verbos, otro con sustantivos, otro con adverbios, y así. https://github.com/florealonso/Words_Clustering/blob/master/cooc%20pos%20triplas%20cluster/wikitriplas_n.ipynb
