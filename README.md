## Projeto de reconhecimento de faces

### O projeto

Trabalho apresentado à disciplina de Aprendizado não Supervisionado, do curso de Pós Graduação em Data Science na instituição de educação FURB, como objetivo de aplicar os conhecimentos aprendidos em aula.

Neste projeto é criado um algorítimo de reconhecimento de faces de pessoas utilizando bibliotecas bastante conhecidas no mundo de data science como o numpy e o pandas, bem como na visão computacional com o OpenCV.

A execução do algorítimo segue as seguintes etapas:

1 - Criação do dataset a partir das imagens armazenadas na pasta "../data/external/ORL2". Nesta pasta estão armazenas 410 imagens, incluído 10 faces de fotos minhas.

2 - Divisão do dataset com 70% para treino e 30% para teste.

3 - Treino do modelo:

    1. Cálculo da imagem média;
    2. Cálculo da diferença das imagens com a média;
    3. Cálculo da matriz de covariância das imagens;
    4. Cálculo dos autovalores e autovetores da matriz de covariância;
    5. Cálculo das eigenfaces;
    6. Cálculo das projeções.
    
4 - Teste do modelo.


Toda a lógica do algorítimo está localizada nas classes criadas na pasta "../src/models".


### Requirementos

- [Python 3.6](https://www.python.org/downloads/windows/) ou superior
- [Jupyter Notebook](https://jupyter.org/install)
- Demais dependências estão contidas no arquivo "requiments.txt" mencionado abaixo no tópico "Como testar o projeto?"

### Como testar o projeto?

1. Baixar o projeto do git;
2. Acessar a pasta raiz do projeto onde o projeto foi armazenado por meior do "propt" e executar o arquivo requirements.txt;

    > $ pip install requirements.txt
    
3. Abrir o jupyter notebook;
3. Acessar o notebook "test-face-recognition_v00" na pasta "../notebooks";
4. Com o notebook "test-face-recognition_v00" aberto, acessar o menu "Cell" e executar o notebook clicando em "Run All".

### Log de saída do algorítmo

```python
pca = PCA(images_path="../data/external/orl2/",
          min_components=10,
          max_components=20,
          test_size=.3)

pca.processing()
pca.result_by_component
```

    UNKNOW - Predicted label: 1, confidence: 1613.8280923704704, reconstructed error: 2617.3421633405137, original label: 16
    UNKNOW - Predicted label: 1, confidence: 1440.2928901518246, reconstructed error: 2617.530324561685, original label: 16
    UNKNOW - Predicted label: 32, confidence: 1957.6171967946223, reconstructed error: 2169.6783171705433, original label: 1
    UNKNOW - Predicted label: 40, confidence: 1433.5126419928106, reconstructed error: 2329.467750367023, original label: 35
    UNKNOW - Predicted label: 15, confidence: 1221.8054464336062, reconstructed error: 2147.5583810457865, original label: 35
    UNKNOW - Predicted label: 22, confidence: 902.1694651831995, reconstructed error: 1951.1442796472022, original label: 39
    UNKNOW PERSON (by two factors) - Predicted label: 21, confidence: 1896.3068827660416, reconstructed error: 2675.4668751453455, original label: 41
    UNKNOW - Predicted label: 35, confidence: 1718.9884129733894, reconstructed error: 2793.468990341579, original label: 41
    UNKNOW - Predicted label: 39, confidence: 1614.9256510200546, reconstructed error: 2609.9262441685973, original label: 41
    Number components: 10, accuracy: 93.5% (114 of 123)
    True positive count: 114, True negative count 1
    Min distance: 2.2250738585072014e-308, Max distance: 1.7976931348623157e+308, Mean distance: 719.1550797912276
    Min rec: 2.2250738585072014e-308, Max rec: 1.7976931348623157e+308, Mean rec: 1949.3109955263183
    Corrects: 114
    UNKNOW - Predicted label: 1, confidence: 1004.1695008966552, reconstructed error: 2018.5427912234113, original label: 16
    UNKNOW - Predicted label: 1, confidence: 1614.032297752676, reconstructed error: 2616.626836214901, original label: 16
    UNKNOW - Predicted label: 1, confidence: 1654.4088673979554, reconstructed error: 2617.784368507078, original label: 16
    UNKNOW - Predicted label: 40, confidence: 1518.8001531289244, reconstructed error: 2189.3926098349743, original label: 35
    UNKNOW - Predicted label: 15, confidence: 1230.81480105106, reconstructed error: 2147.3097121747483, original label: 35
    UNKNOW - Predicted label: 22, confidence: 921.699936343458, reconstructed error: 1947.6054528574314, original label: 39
    UNKNOW PERSON (by two factors) - Predicted label: 21, confidence: 2003.3024902060808, reconstructed error: 2616.6277916432823, original label: 41
    UNKNOW PERSON (by two factors) - Predicted label: 35, confidence: 1865.589984302171, reconstructed error: 2779.5816231943973, original label: 41
    UNKNOW - Predicted label: 39, confidence: 1668.314535217914, reconstructed error: 2569.922372368473, original label: 41
    Number components: 11, accuracy: 94.31% (114 of 123)
    True positive count: 114, True negative count 2
    Min distance: 2.2250738585072014e-308, Max distance: 1.7976931348623157e+308, Mean distance: 1474.3456200938983
    Min rec: 2.2250738585072014e-308, Max rec: 1.7976931348623157e+308, Mean rec: 3865.51516881305
    Corrects: 228
    UNKNOW - Predicted label: 1, confidence: 1007.6516865443563, reconstructed error: 2016.4555536882037, original label: 16
    UNKNOW - Predicted label: 1, confidence: 1614.386764394662, reconstructed error: 2616.307130288797, original label: 16
    UNKNOW - Predicted label: 1, confidence: 1654.4570556777082, reconstructed error: 2617.5967985921743, original label: 16
    UNKNOW - Predicted label: 15, confidence: 1716.3783411795846, reconstructed error: 2142.844371390512, original label: 35
    UNKNOW - Predicted label: 15, confidence: 1291.0032341521053, reconstructed error: 2129.1087337193467, original label: 35
    UNKNOW - Predicted label: 29, confidence: 1278.625943290881, reconstructed error: 1982.126131203562, original label: 39
    UNKNOW - Predicted label: 22, confidence: 972.304765206563, reconstructed error: 1937.960525913776, original label: 39
    UNKNOW PERSON (by two factors) - Predicted label: 4, confidence: 2042.667658965702, reconstructed error: 2602.852857923398, original label: 41
    UNKNOW PERSON (by two factors) - Predicted label: 35, confidence: 2056.132216201955, reconstructed error: 2774.1279710928984, original label: 41
    UNKNOW - Predicted label: 39, confidence: 1684.447485247148, reconstructed error: 2568.956013636668, original label: 41
    Number components: 12, accuracy: 93.5% (113 of 123)
    True positive count: 113, True negative count 2
    Min distance: 2.2250738585072014e-308, Max distance: 1.7976931348623157e+308, Mean distance: 2251.428339161279
    Min rec: 2.2250738585072014e-308, Max rec: 1.7976931348623157e+308, Mean rec: 5753.05553097057
    Corrects: 341
    UNKNOW - Predicted label: 1, confidence: 1643.941148038066, reconstructed error: 2613.328720234024, original label: 16
    UNKNOW - Predicted label: 1, confidence: 1677.0515425376886, reconstructed error: 2615.984136037526, original label: 16
    UNKNOW - Predicted label: 15, confidence: 1719.5450523205313, reconstructed error: 2142.419426722975, original label: 35
    UNKNOW - Predicted label: 15, confidence: 1320.584783376452, reconstructed error: 2116.134211244646, original label: 35
    UNKNOW - Predicted label: 29, confidence: 1288.4107995408503, reconstructed error: 1980.1219659404821, original label: 39
    UNKNOW - Predicted label: 22, confidence: 1031.974476950642, reconstructed error: 1919.55411489231, original label: 39
    UNKNOW PERSON (by two factors) - Predicted label: 4, confidence: 2070.590321814152, reconstructed error: 2551.834438203231, original label: 41
    UNKNOW PERSON (by two factors) - Predicted label: 35, confidence: 2065.4030794422324, reconstructed error: 2727.5179192811916, original label: 41
    UNKNOW - Predicted label: 39, confidence: 1684.892223809212, reconstructed error: 2566.5449538241096, original label: 41
    Number components: 13, accuracy: 94.31% (114 of 123)
    True positive count: 114, True negative count 2
    Min distance: 2.2250738585072014e-308, Max distance: 1.7976931348623157e+308, Mean distance: 3055.7327182997656
    Min rec: 2.2250738585072014e-308, Max rec: 1.7976931348623157e+308, Mean rec: 7616.164453207446
    Corrects: 455
    UNKNOW PERSON (by two factors) - Predicted label: 1, confidence: 1880.3089256957805, reconstructed error: 2604.1833652797955, original label: 16
    UNKNOW PERSON (by two factors) - Predicted label: 1, confidence: 1883.9546842376512, reconstructed error: 2603.405270026163, original label: 16
    UNKNOW - Predicted label: 15, confidence: 1456.9087363679832, reconstructed error: 2251.5014990001673, original label: 32
    UNKNOW - Predicted label: 15, confidence: 1722.3597579387144, reconstructed error: 2132.1428188561854, original label: 35
    UNKNOW - Predicted label: 15, confidence: 1325.6915437059865, reconstructed error: 2116.4989959836976, original label: 35
    UNKNOW - Predicted label: 29, confidence: 1298.3961446852084, reconstructed error: 1958.4128778171369, original label: 39
    UNKNOW - Predicted label: 22, confidence: 1034.6864915437468, reconstructed error: 1919.9557286562626, original label: 39
    UNKNOW PERSON (by two factors) - Predicted label: 4, confidence: 2087.397631365138, reconstructed error: 2548.5882366518135, original label: 41
    UNKNOW PERSON (by two factors) - Predicted label: 35, confidence: 2065.4839987837804, reconstructed error: 2726.0715324437106, original label: 41
    UNKNOW - Predicted label: 39, confidence: 1742.5372781116644, reconstructed error: 2554.6847163593397, original label: 41
    Number components: 14, accuracy: 95.12% (113 of 123)
    True positive count: 113, True negative count 4
    Min distance: 2.2250738585072014e-308, Max distance: 1.7976931348623157e+308, Mean distance: 3893.146199683727
    Min rec: 2.2250738585072014e-308, Max rec: 1.7976931348623157e+308, Mean rec: 9457.350018110328
    Corrects: 568
    UNKNOW PERSON (by two factors) - Predicted label: 1, confidence: 1975.1892369584452, reconstructed error: 2587.3954085141295, original label: 16
    UNKNOW PERSON (by two factors) - Predicted label: 1, confidence: 1914.2686885747912, reconstructed error: 2586.116973379201, original label: 16
    UNKNOW - Predicted label: 15, confidence: 1459.1630525550254, reconstructed error: 2249.709981308702, original label: 32
    UNKNOW - Predicted label: 15, confidence: 1734.1349167991732, reconstructed error: 2119.048135366443, original label: 35
    UNKNOW - Predicted label: 15, confidence: 1332.4925531530814, reconstructed error: 2096.727450099321, original label: 35
    UNKNOW - Predicted label: 22, confidence: 1040.296527063821, reconstructed error: 1903.1082470526999, original label: 39
    UNKNOW PERSON (by two factors) - Predicted label: 4, confidence: 2088.752842330597, reconstructed error: 2548.7604830583828, original label: 41
    UNKNOW PERSON (by two factors) - Predicted label: 35, confidence: 2324.68868591396, reconstructed error: 2627.6879571212407, original label: 41
    UNKNOW - Predicted label: 39, confidence: 1784.697291929711, reconstructed error: 2545.7040283583638, original label: 41
    Number components: 15, accuracy: 95.93% (114 of 123)
    True positive count: 114, True negative count 4
    Min distance: 2.2250738585072014e-308, Max distance: 1.7976931348623157e+308, Mean distance: 4748.875764023856
    Min rec: 2.2250738585072014e-308, Max rec: 1.7976931348623157e+308, Mean rec: 11274.915489894911
    Corrects: 682
    UNKNOW PERSON (by two factors) - Predicted label: 1, confidence: 1991.9397632438215, reconstructed error: 2565.153796558795, original label: 16
    UNKNOW PERSON (by two factors) - Predicted label: 1, confidence: 2035.6304342702053, reconstructed error: 2548.2943315088232, original label: 16
    UNKNOW - Predicted label: 15, confidence: 1491.0176087555146, reconstructed error: 2224.7748200660667, original label: 32
    UNKNOW - Predicted label: 15, confidence: 1828.3013888302999, reconstructed error: 2108.305006397319, original label: 35
    UNKNOW - Predicted label: 15, confidence: 1337.7519757340565, reconstructed error: 2091.917302380761, original label: 35
    UNKNOW - Predicted label: 22, confidence: 1137.745762134198, reconstructed error: 1855.4142394624441, original label: 39
    UNKNOW PERSON (by two factors) - Predicted label: 4, confidence: 2091.2721884388448, reconstructed error: 2546.493667771432, original label: 41
    UNKNOW PERSON (by two factors) - Predicted label: 35, confidence: 2407.2357115107884, reconstructed error: 2531.4671635239515, original label: 41
    UNKNOW - Predicted label: 39, confidence: 1784.701693927085, reconstructed error: 2538.288399689838, original label: 41
    Number components: 16, accuracy: 95.93% (114 of 123)
    True positive count: 114, True negative count 4
    Min distance: 2.2250738585072014e-308, Max distance: 1.7976931348623157e+308, Mean distance: 5624.355550382702
    Min rec: 2.2250738585072014e-308, Max rec: 1.7976931348623157e+308, Mean rec: 13072.603634721825
    Corrects: 796
    UNKNOW PERSON (by two factors) - Predicted label: 1, confidence: 2031.0700016210128, reconstructed error: 2529.738721686491, original label: 16
    UNKNOW PERSON (by two factors) - Predicted label: 1, confidence: 2042.3052613425918, reconstructed error: 2515.9226935659212, original label: 16
    UNKNOW - Predicted label: 15, confidence: 1498.373227294115, reconstructed error: 2224.579735590523, original label: 32
    UNKNOW - Predicted label: 15, confidence: 1834.0459692850875, reconstructed error: 2104.784311990186, original label: 35
    UNKNOW - Predicted label: 15, confidence: 1370.5706552096458, reconstructed error: 2053.5362183316856, original label: 35
    UNKNOW - Predicted label: 29, confidence: 1343.6824010483663, reconstructed error: 1872.8969539192485, original label: 39
    UNKNOW - Predicted label: 22, confidence: 1238.344452404724, reconstructed error: 1822.9914975117135, original label: 39
    UNKNOW PERSON (by two factors) - Predicted label: 4, confidence: 2094.2404342693562, reconstructed error: 2543.325578843574, original label: 41
    UNKNOW PERSON (by two factors) - Predicted label: 35, confidence: 2407.7326299963315, reconstructed error: 2501.7377960130034, original label: 41
    UNKNOW PERSON (by two factors) - Predicted label: 39, confidence: 1894.6858814711845, reconstructed error: 2531.5112087446896, original label: 41
    Number components: 17, accuracy: 95.93% (113 of 123)
    True positive count: 113, True negative count 5
    Min distance: 2.2250738585072014e-308, Max distance: 1.7976931348623157e+308, Mean distance: 6517.590618736936
    Min rec: 2.2250738585072014e-308, Max rec: 1.7976931348623157e+308, Mean rec: 14851.873239696715
    Corrects: 909
    UNKNOW PERSON (by two factors) - Predicted label: 1, confidence: 2060.3463467329802, reconstructed error: 2429.8547281679207, original label: 16
    UNKNOW PERSON (by two factors) - Predicted label: 1, confidence: 2103.565587614504, reconstructed error: 2434.2255441926495, original label: 16
    UNKNOW - Predicted label: 15, confidence: 1498.4447346500522, reconstructed error: 2222.7851898012996, original label: 32
    UNKNOW - Predicted label: 15, confidence: 1837.5554889761252, reconstructed error: 2076.4127720662864, original label: 35
    UNKNOW - Predicted label: 15, confidence: 1483.3996887265203, reconstructed error: 1942.1820203060267, original label: 35
    UNKNOW - Predicted label: 5, confidence: 1087.6992337858735, reconstructed error: 1539.2040150675284, original label: 40
    UNKNOW PERSON (by two factors) - Predicted label: 4, confidence: 2156.6677301513782, reconstructed error: 2543.5772054333243, original label: 41
    UNKNOW PERSON (by two factors) - Predicted label: 35, confidence: 2424.0984578466696, reconstructed error: 2491.0313125290095, original label: 41
    UNKNOW PERSON (by two factors) - Predicted label: 21, confidence: 2007.47313982935, reconstructed error: 2482.0519736701726, original label: 41
    Number components: 18, accuracy: 96.75% (114 of 123)
    True positive count: 114, True negative count 5
    Min distance: 2.2250738585072014e-308, Max distance: 1.7976931348623157e+308, Mean distance: 7429.348187564305
    Min rec: 2.2250738585072014e-308, Max rec: 1.7976931348623157e+308, Mean rec: 16611.935101734944
    Corrects: 1023
    UNKNOW PERSON (by two factors) - Predicted label: 1, confidence: 2113.0284735842843, reconstructed error: 2403.7709957481393, original label: 16
    UNKNOW PERSON (by two factors) - Predicted label: 1, confidence: 2139.777647224482, reconstructed error: 2410.196465021057, original label: 16
    UNKNOW - Predicted label: 15, confidence: 1541.3059042703378, reconstructed error: 2222.060080195853, original label: 32
    UNKNOW - Predicted label: 15, confidence: 1905.9722897418874, reconstructed error: 2020.1668247944276, original label: 35
    UNKNOW - Predicted label: 15, confidence: 1483.4734478100042, reconstructed error: 1941.8967531771611, original label: 35
    UNKNOW - Predicted label: 29, confidence: 1417.0569188457143, reconstructed error: 1838.824080764661, original label: 39
    UNKNOW - Predicted label: 5, confidence: 1096.261712095173, reconstructed error: 1539.1851740450204, original label: 40
    UNKNOW PERSON (by two factors) - Predicted label: 4, confidence: 2157.071358810045, reconstructed error: 2534.128055170062, original label: 41
    UNKNOW PERSON (by distance) - Predicted label: 35, confidence: 2504.167474294672, reconstructed error: 2396.202620814859, original label: 41
    UNKNOW PERSON (by two factors) - Predicted label: 21, confidence: 2016.1091157066735, reconstructed error: 2472.666374584327, original label: 41
    Number components: 19, accuracy: 95.93% (113 of 123)
    True positive count: 113, True negative count 5
    Min distance: 2.2250738585072014e-308, Max distance: 1.7976931348623157e+308, Mean distance: 8357.590092381648
    Min rec: 2.2250738585072014e-308, Max rec: 1.7976931348623157e+308, Mean rec: 18354.504491602638
    Corrects: 1136
    UNKNOW PERSON (by two factors) - Predicted label: 1, confidence: 2148.5299169790283, reconstructed error: 2404.1256206779212, original label: 16
    UNKNOW PERSON (by two factors) - Predicted label: 1, confidence: 2141.497340095963, reconstructed error: 2409.8211551897375, original label: 16
    UNKNOW - Predicted label: 15, confidence: 1547.5436906342434, reconstructed error: 2221.6556888951086, original label: 32
    UNKNOW - Predicted label: 15, confidence: 1906.165630385473, reconstructed error: 2014.774925394894, original label: 35
    UNKNOW - Predicted label: 15, confidence: 1522.2602496894597, reconstructed error: 1920.6975295449306, original label: 35
    UNKNOW - Predicted label: 29, confidence: 1423.5295424241096, reconstructed error: 1839.156872047624, original label: 39
    UNKNOW - Predicted label: 5, confidence: 1132.0720791469093, reconstructed error: 1512.711803351848, original label: 40
    UNKNOW PERSON (by two factors) - Predicted label: 4, confidence: 2181.9671300030504, reconstructed error: 2529.482555780925, original label: 41
    UNKNOW PERSON (by distance) - Predicted label: 35, confidence: 2574.1992210846483, reconstructed error: 2359.4456976162855, original label: 41
    UNKNOW PERSON (by two factors) - Predicted label: 21, confidence: 2032.8895655107444, reconstructed error: 2472.17090833138, original label: 41
    Number components: 20, accuracy: 95.93% (113 of 123)
    True positive count: 113, True negative count 5
    Min distance: 2.2250738585072014e-308, Max distance: 1.7976931348623157e+308, Mean distance: 9300.185228826396
    Min rec: 2.2250738585072014e-308, Max rec: 1.7976931348623157e+308, Mean rec: 20080.290283142855
    Corrects: 1249
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Components</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>93.50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>94.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>93.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>94.31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>95.12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15</td>
      <td>95.93</td>
    </tr>
    <tr>
      <th>6</th>
      <td>16</td>
      <td>95.93</td>
    </tr>
    <tr>
      <th>7</th>
      <td>17</td>
      <td>95.93</td>
    </tr>
    <tr>
      <th>8</th>
      <td>18</td>
      <td>96.75</td>
    </tr>
    <tr>
      <th>9</th>
      <td>19</td>
      <td>95.93</td>
    </tr>
    <tr>
      <th>10</th>
      <td>20</td>
      <td>95.93</td>
    </tr>
  </tbody>
</table>
</div>

