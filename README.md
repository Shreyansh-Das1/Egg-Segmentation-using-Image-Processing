# Egg-Segmentation-using-Image-Processing

**Task-1: Segmentation using Pre-trained Model (FastSAM)**

* Using prompt-based segmentation caused a background confusion:  
  ![][image1]  
  *Figure 1\. Prompt-based segmentation; Hand was segmented instead of egg*  
* Moving to Geometry based segmentation, i.e, w/o prompt, we will be considering Elliptical Eccentricity as a criteria. According to \[1\], the Eccentricity ratio of Eggs varies from 0.5256 to 0.7671.  
  Eccentricity Ratio refers to how close a polygon is to being a perfect circle, where 0 being a perfect circle.  
  ![][image2]![][image3]  
  *Figure 2\. Eccentricity Ratio \-based segmentation*  
    
  However, in other images, the same failed to properly segment the egg, and incorrectly segmented the tip of a finger instead as shown in Figure 3\.  
    
  ![][image4]![][image5]  
  *Figure 3\. Failed segmentation*  
* As a workaround, The Eccentricity ratio was replaced with other parameters such as Convexity Ratio (as an egg is technically convex shaped), Aspect Ratio\[2\] (ratio of the longest length with the shortest), and De-Duplication, as FastSAM creates multiple overlapping masks for the same object, any masks having IoU over 0.5 will be merged into one.  
* The Result, Objects were segmented in about 90% of the total images, sample Images are shown in *Figure 4*:

  *Found 108 images to process.*

  *\[11/108\] Processing Not Damaged (108).jpg... No eggs identified.*

  *\[17/108\] Processing Not Damaged (16).jpg... No eggs identified.*

  *\[19/108\] Processing Not Damaged (18).jpg... No eggs identified.*

  *\[20/108\] Processing Not Damaged (19).jpg... No eggs identified.*

  *\[24/108\] Processing Not Damaged (22).jpg... No eggs identified.*

  *\[25/108\] Processing Not Damaged (23).jpg... No eggs identified.*

  *\[57/108\] Processing Not Damaged (52).jpg... No eggs identified.*

  *\[70/108\] Processing Not Damaged (64).jpg... No eggs identified.*

  *\[79/108\] Processing Not Damaged (72).jpg... No eggs identified.*

  *\[107/108\] Processing Not Damaged (98).jpg... No eggs identified.*


  *Processing complete. Masks stored in: /kaggle/working/egg\_masks*

  *![][image6]*

  *Figure 4\. Geometrically filtered segmentation*


* Unfortunately, background was being segmented as well. Therefore I had to consult AI for suggestions. The reason and suggestion provided was: 

  *The failure modes visible in your mask thumbnails (e.g., masks (5), (20), (26), (49), (50)) stem from three specific issues in your current script:*

1. ***FastSAM without prompts acts as an "everything" segmenter:** It segments background tiles, walls, fingers, palms, and table corners.*  
2. ***Hand/Finger Merging:** FastSAM groups fingers and thumb skin touching the egg into a single component, creating non-convex blobs (e.g., (14), (25), (29)).*  
3. ***Geometric Heuristics are Leaky:** Background clutter (squares, wall seams, palm segments) happens to pass solidity \> 0.88 and aspect ratio 1.05–1.8.*

   *Rather than fine-tuning classical geometric filters on top of blind masks, use a **two-stage zero-shot pipeline**: **OWLv2 / Grounding DINO** (to propose an egg bounding box) $\rightarrow$ **SAM / FastSAM prompted with that box**.*

* Upon Implementing the suggestion, no eggs were detected in 13 images compared to 10 in Geometric Filtering, leaving 95 masks out of 108\. The remaining 13 image will have to be deleted before moving to Task-2, (Or maybe not incase Task-2 outperforms Task-1)

![][image7]  
*Figure 5\. Result using 2-stage zero shot segmentation*

* **Note:-**  As per the Assessment document, I hereby disclose the usage of AI in the assessment in the following manner:   
  * I decided the approach and used Gemini Flask for the implementation, and debugging, while understanding each line.  
  * It suggested to use Aspect ratio instead of eccentricity ratio due to the following reasoning when *figure 2* occured: 

    ***In Practice:** While both can be used to filter shapes, **Aspect Ratio** is generally preferred in image processing code because it provides a direct linear intuition of proportions without the square root overhead.* 

**Task-2: Segmentation using Image Processing**

1. **Trial-1**   
   For Task-2, I will be exploring Canny Edge Detection for initially segmenting everything, followed by Otsu Threshold for Binary mask, and lastly, Aspect Ratio for egg detection, along with scale, and convexity.

The result is in *figure 6\.*   
*![][image8]*  
*Figure 6: Using Canny \+ Otsu \+ Aspect Ratio*  
This approach gave little to no  helpful segmentation in any manner; The reason which AI gave was:

### ***Root-Cause Diagnostics of the Black Masks***

1. ***The Inversion Bug (if np.mean(otsu\_mask) \> 127:):***  
   * *Handheld egg images feature a large field of view of hands, skin, and table surfaces, with the egg occupying 5–20% of the image.*  
   * *When Otsu thresholds the image, skin and egg are often both brighter than dark shadows, or the table is bright, resulting in an unpredictable global mean intensity. The global inversion check flips the mask arbitrarily, making the foreground 0 and turning the downstream pipeline into pure black.*  
2. ***Contour Leakage via cv2.bitwise\_or(otsu\_mask, edge\_filled):***  
   * *Otsu on grayscale merges the egg with pale skin (fingers/palm) into one connected giant blob.*  
   * *This combined blob violates area \< 0.40 \* img\_area and solidity \>= 0.85, so the filter drops the egg entirely.*  
3. ***Canny Disconnection:***  
   * *Canny edges have micro-gaps around subtle lighting transitions. A $7\times 7$ morphological closing failed to bridge the gaps, so cv2.drawContours(..., thickness=cv2.FILLED) filled nothing.*  
4. ***Circularity & Ellipse Filter Dropouts:***  
   * *Partially occluded eggs or eggs viewed at an angle have circularity $<0.55$, causing immediate rejection.*

**B. Watershed Method:** Next watershed method will be explored as an alternative.

 A massive improvement is visible compared to trial-1, however , some images are still complete black or have noise/incorrect segmentation![][image9]

***Figure 7: Using Watershed***

**C. AI-Suggested Pipeline:**

The pipeline starts with the decoupling of object boundary and background noise through bilateral filtering, which eliminates surface textures while retaining sharpness of edge outer shell. At the same time, the image is cast into multiple colour spaces $YCrCb$, $HSV$ and $Lab$ to suppress human skin and extract a high-contrast binary candidate mask based on eggshell reflectance. After the candidate mask is obtained, it is first cleaned with border-clearing operations to remove the edges of the table (or clothing) from the mask. The cleaned mask is also subjected to a morphological opening and closing to remove any thin skin bridges and fill in the internal specular highlights. Then, in the case of clustered or touching eggs, the mask will be split into topological seeds using a Euclidean distance transform with an adaptive threshold. This process identifies the deep interior centroid of each egg while also determining sure-background regions through dilation. The marker-controlled watershed algorithm uses those seed markers to flood topographic image gradients from inside to outside to separate all egg instances at their correct physical boundaries. In the end, the source contours are passed through geometric validation filters that check for scale, high convex solidity, circularity, and elliptical aspect ratio to prune the remaining false positives and yield final segmented egg masks. 

**![][image10]**

***Figure 8; AI Suggested Mask***

**Task-3: Comparative Analysis of Task-2 with Task-1**  
To compare the 2 trials of Task-2 with respect to Task-1 ( which is assumed to be the ground truth masks),  
We will be comparing both based on IoU, and Pixel Accuracy, the result of which is in *table 1* and *table 2* respectively.

|  | *Mean IoU* | *Standard IoU* |  *Mean Pixel Accuracy* | *Mean Dice* |
| :---- | :---- | :---- | :---- | :---- |
| *Canny \+ Otsu+ Aspect Ratio* | *0.0572* | *0.1504* | *0.565* | *0.0805* |
| *Watershed* | *0.254769134* | *0.284326575* | *0.641599892* | *0.334278801* |
| *Ai-Suggested Algorithm* | *0.129107191* | *0.235110112* | *0.581479621* | *0.171084169* |

*Table 1\. IoU (Intersection over Union) Performance*

| *Method Pipeline* | *Mean Pixel Acc (%)* | *Median Pixel Acc (%)* | *Min Pixel Acc (%)* | *Complete Dropouts (IoU=0)* | *High Overlap (IoU â‰¥ 0.70)* |
| :---- | :---- | :---- | :---- | :---- | :---- |
| *Otsu \+ Canny* | *56.49681975* | *61.8996582* | *8.198486328* | *61 / 76 (80.3%)* | *1 / 76 (1.3%)* |
| *Watershed* | *64.15998921* | *71.07873535* | *8.198486328* | *28 / 76 (36.8%)* | *9 / 76 (11.8%)* |
| *AI-suggested Algoritum* | *58.14796207* | *65.0793457* | *8.198486328* | *52 / 76 (68.4%)* | *4 / 76 (5.3%)* |

*Table 2\. Pixel Accuracy*  
As per *table 1* and *table 2,* the Watershed algorithm outperformed the rest. That could be an algorithm to improve upon  
**References**

* ***A Quick Guide on Segmentation:** [https\://www\.geeksforgeeks.org/computer-vision/image-segmentation-models/](https://www.geeksforgeeks.org/computer-vision/image-segmentation-models/)*

*\[1\]	N. Jia, B. Li, et al Exploratory Study of Sex Identification for Chicken Embryos Based on Blood Vessel Images and Deep Learning; [https\://doi.org/10.3390/agriculture13081480](https://doi.org/10.3390/agriculture13081480)*  
*\[2\]	V. Narushin, et al. Egg Geometrical Index: Encompassing a wide range of avian egg profiles with potential for novel AI applications in research and industry; https\://doi.org/10.1016/j.jfca.2025.108143*

