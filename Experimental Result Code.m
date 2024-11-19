%% RGB image convert to YCbCr image
A = imread("OK\img_000000333.jpg");
B = imread("OK\img_000000200 (2).jpg");
C = imread("Not OK\img_00000064.jpg");
D = imread("Not OK\img_00000005.jpg");

ycbcr_image_A = rgb2ycbcr(A); 
ycbcr_image_B = rgb2ycbcr(B); 
ycbcr_image_C = rgb2ycbcr(C); 
ycbcr_image_D = rgb2ycbcr(D); 

figure, 
subplot(2,4,1); imshow(A); title('Compliance');
subplot(2,4,2); imshow(B); title('Compliance');
subplot(2,4,3); imshow(C); title('Non-compliance');
subplot(2,4,4); imshow(D); title('Non-compliance');
subplot(2,4,5); imshow(ycbcr_image_A); title('Compliance');
subplot(2,4,6); imshow(ycbcr_image_B); title('Compliance');
subplot(2,4,7); imshow(ycbcr_image_C); title('Non-compliance');
subplot(2,4,8); imshow(ycbcr_image_D); title('Non-compliance');
%% Extract out the Y channel of YCbCr image
Y_A = ycbcr_image_A(:,:,1);
Y_B = ycbcr_image_B(:,:,1);
Y_C = ycbcr_image_C(:,:,1);
Y_D = ycbcr_image_D(:,:,1);

figure, 
subplot(1,4,1); imshow(Y_A); title('Compliance');
subplot(1,4,2); imshow(Y_B); title('Compliance');
subplot(1,4,3); imshow(Y_C); title('Non-compliance');
subplot(1,4,4); imshow(Y_D); title('Non-compliance');
%% Apply CLAHE to the Y channel
clahe_Y_A = adapthisteq(Y_A);
clahe_Y_B = adapthisteq(Y_B);
clahe_Y_C = adapthisteq(Y_C);
clahe_Y_D = adapthisteq(Y_D);

figure, 
subplot(1,4,1); imshow(clahe_Y_A); title('Compliance');
subplot(1,4,2); imshow(clahe_Y_B); title('Compliance');
subplot(1,4,3); imshow(clahe_Y_C); title('Non-compliance');
subplot(1,4,4); imshow(clahe_Y_D); title('Non-compliance');
%% Placed back modified Y channel to the YCbCr image
ycbcr_image_A(:,:,1) = clahe_Y_A;
ycbcr_image_B(:,:,1) = clahe_Y_B;
ycbcr_image_C(:,:,1) = clahe_Y_C;
ycbcr_image_D(:,:,1) = clahe_Y_D;

% Assign preprocessed image as YCbCr image
preprocessed_img_A = ycbcr_image_A;
preprocessed_img_B = ycbcr_image_B;
preprocessed_img_C = ycbcr_image_C;
preprocessed_img_D = ycbcr_image_D;

figure, 
subplot(1,4,1); imshow(preprocessed_img_A); title('Compliance');
subplot(1,4,2); imshow(preprocessed_img_B); title('Compliance');
subplot(1,4,3); imshow(preprocessed_img_C); title('Non-compliance');
subplot(1,4,4); imshow(preprocessed_img_D); title('Non-compliance');
%% Skin Segmentation
lower_skin = [0, 77, 133];  
upper_skin = [150, 127, 173];

skin_mask_A = (preprocessed_img_A(:,:,2) >= lower_skin(2)) & (preprocessed_img_A(:,:,2) <= upper_skin(2)) & ...
              (preprocessed_img_A(:,:,3) >= lower_skin(3)) & (preprocessed_img_A(:,:,3) <= upper_skin(3));

skin_mask_B = (preprocessed_img_B(:,:,2) >= lower_skin(2)) & (preprocessed_img_B(:,:,2) <= upper_skin(2)) & ...
              (preprocessed_img_B(:,:,3) >= lower_skin(3)) & (preprocessed_img_B(:,:,3) <= upper_skin(3));

skin_mask_C = (preprocessed_img_C(:,:,2) >= lower_skin(2)) & (preprocessed_img_C(:,:,2) <= upper_skin(2)) & ...
              (preprocessed_img_C(:,:,3) >= lower_skin(3)) & (preprocessed_img_C(:,:,3) <= upper_skin(3));

skin_mask_D = (preprocessed_img_D(:,:,2) >= lower_skin(2)) & (preprocessed_img_D(:,:,2) <= upper_skin(2)) & ...
              (preprocessed_img_D(:,:,3) >= lower_skin(3)) & (preprocessed_img_D(:,:,3) <= upper_skin(3));

figure, 
subplot(1,4,1); imshow(skin_mask_A); title('Compliance');
subplot(1,4,2); imshow(skin_mask_B); title('Compliance');
subplot(1,4,3); imshow(skin_mask_C); title('Non-compliance');
subplot(1,4,4); imshow(skin_mask_D); title('Non-compliance');
%% morphological
skin_mask_A = imopen(skin_mask_A, strel('disk', 6));
skin_mask_B = imopen(skin_mask_B, strel('disk', 6));
skin_mask_C = imopen(skin_mask_C, strel('disk', 6));
skin_mask_D = imopen(skin_mask_D, strel('disk', 6));

figure, 
subplot(1,4,1); imshow(skin_mask_A); title('Compliance');
subplot(1,4,2); imshow(skin_mask_B); title('Compliance');
subplot(1,4,3); imshow(skin_mask_C); title('Non-compliance');
subplot(1,4,4); imshow(skin_mask_D); title('Non-compliance');
%% Non-skin areas are set to black
skin_segmented_A = A; 
skin_segmented_A(repmat(~skin_mask_A, [1 1 3])) = 0;  
skin_segmented_B = B; 
skin_segmented_B(repmat(~skin_mask_B, [1 1 3])) = 0; 
skin_segmented_C = C; 
skin_segmented_C(repmat(~skin_mask_C, [1 1 3])) = 0;  
skin_segmented_D = D; 
skin_segmented_D(repmat(~skin_mask_D, [1 1 3])) = 0;  

figure, 
subplot(1,4,1); imshow(skin_segmented_A); title('Compliance');
subplot(1,4,2); imshow(skin_segmented_B); title('Compliance');
subplot(1,4,3); imshow(skin_segmented_C); title('Non-compliance');
subplot(1,4,4); imshow(skin_segmented_D); title('Non-compliance');
%% Convert to HSV images for clothing segmentation
hsv_image_A = rgb2hsv(A);
hsv_image_B = rgb2hsv(B);
hsv_image_C = rgb2hsv(C);
hsv_image_D = rgb2hsv(D);

lower_cloth = [0, 0, 0];
upper_cloth = [1, 1, 0.4];

clothing_mask_A = (hsv_image_A(:,:,1) >= lower_cloth(1)) & (hsv_image_A(:,:,1) <= upper_cloth(1)) & ...
                  (hsv_image_A(:,:,2) >= lower_cloth(2)) & (hsv_image_A(:,:,2) <= upper_cloth(2)) & ...
                  (hsv_image_A(:,:,3) >= lower_cloth(3)) & (hsv_image_A(:,:,3) <= upper_cloth(3));

clothing_mask_B = (hsv_image_B(:,:,1) >= lower_cloth(1)) & (hsv_image_B(:,:,1) <= upper_cloth(1)) & ...
                  (hsv_image_B(:,:,2) >= lower_cloth(2)) & (hsv_image_B(:,:,2) <= upper_cloth(2)) & ...
                  (hsv_image_B(:,:,3) >= lower_cloth(3)) & (hsv_image_B(:,:,3) <= upper_cloth(3));

clothing_mask_C = (hsv_image_C(:,:,1) >= lower_cloth(1)) & (hsv_image_C(:,:,1) <= upper_cloth(1)) & ...
                  (hsv_image_C(:,:,2) >= lower_cloth(2)) & (hsv_image_C(:,:,2) <= upper_cloth(2)) & ...
                  (hsv_image_C(:,:,3) >= lower_cloth(3)) & (hsv_image_C(:,:,3) <= upper_cloth(3));

clothing_mask_D = (hsv_image_D(:,:,1) >= lower_cloth(1)) & (hsv_image_D(:,:,1) <= upper_cloth(1)) & ...
                  (hsv_image_D(:,:,2) >= lower_cloth(2)) & (hsv_image_D(:,:,2) <= upper_cloth(2)) & ...
                  (hsv_image_D(:,:,3) >= lower_cloth(3)) & (hsv_image_D(:,:,3) <= upper_cloth(3));

figure, 
subplot(2,4,1); imshow(hsv_image_A); title('Compliance');
subplot(2,4,2); imshow(hsv_image_B); title('Compliance');
subplot(2,4,3); imshow(hsv_image_C); title('Non-compliance');
subplot(2,4,4); imshow(hsv_image_D); title('Non-compliance');
subplot(2,4,5); imshow(clothing_mask_A); title('Compliance');
subplot(2,4,6); imshow(clothing_mask_B); title('Compliance');
subplot(2,4,7); imshow(clothing_mask_C); title('Non-compliance');
subplot(2,4,8); imshow(clothing_mask_D); title('Non-compliance');
%% Remove skin areas from the clothing mask
clothing_mask_A = clothing_mask_A & ~skin_mask_A;
clothing_mask_B = clothing_mask_B & ~skin_mask_B;
clothing_mask_C = clothing_mask_C & ~skin_mask_C;
clothing_mask_D = clothing_mask_D & ~skin_mask_D;

figure, 
subplot(1,4,1); imshow(clothing_mask_A); title('Compliance');
subplot(1,4,2); imshow(clothing_mask_B); title('Compliance');
subplot(1,4,3); imshow(clothing_mask_C); title('Non-compliance');
subplot(1,4,4); imshow(clothing_mask_D); title('Non-compliance');
%% Clean the clothing mask using morphological operations
clothing_mask_A = imclose(clothing_mask_A, strel('disk', 6));
clothing_mask_B = imclose(clothing_mask_B, strel('disk', 6));
clothing_mask_C = imclose(clothing_mask_C, strel('disk', 6));
clothing_mask_D = imclose(clothing_mask_D, strel('disk', 6));

figure, 
subplot(1,4,1); imshow(clothing_mask_A); title('Compliance');
subplot(1,4,2); imshow(clothing_mask_B); title('Compliance');
subplot(1,4,3); imshow(clothing_mask_C); title('Non-compliance');
subplot(1,4,4); imshow(clothing_mask_D); title('Non-compliance');
%% Non-clothing areas set to black
clothing_segmented_A = A;
clothing_segmented_A(repmat(~clothing_mask_A, [1 1 3])) = 0;
clothing_segmented_B = B;
clothing_segmented_B(repmat(~clothing_mask_B, [1 1 3])) = 0;
clothing_segmented_C = C;
clothing_segmented_C(repmat(~clothing_mask_C, [1 1 3])) = 0;
clothing_segmented_D = D;
clothing_segmented_D(repmat(~clothing_mask_D, [1 1 3])) = 0;

figure, 
subplot(1,4,1); imshow(clothing_segmented_A); title('Compliance');
subplot(1,4,2); imshow(clothing_segmented_B); title('Compliance');
subplot(1,4,3); imshow(clothing_segmented_C); title('Non-compliance');
subplot(1,4,4); imshow(clothing_segmented_D); title('Non-compliance');
%% Combine skin and clothing masks and set all non-skin and non-clothing areas to white
combined_mask_A = ~skin_mask_A | clothing_mask_A;
combined_mask_B = ~skin_mask_B | clothing_mask_B;
combined_mask_C = ~skin_mask_C | clothing_mask_C;
combined_mask_D = ~skin_mask_D | clothing_mask_D;

final_image_A = A;
final_image_A(repmat(~combined_mask_A, [1 1 3])) = 255;

final_image_B = B;
final_image_B(repmat(~combined_mask_B, [1 1 3])) = 255;

final_image_C = C;
final_image_C(repmat(~combined_mask_C, [1 1 3])) = 255;

final_image_D = D;
final_image_D(repmat(~combined_mask_D, [1 1 3])) = 255;

figure, 
subplot(2,4,1); imshow(combined_mask_A); title('Compliance');
subplot(2,4,2); imshow(combined_mask_B); title('Compliance');
subplot(2,4,3); imshow(combined_mask_C); title('Non-compliance');
subplot(2,4,4); imshow(combined_mask_D); title('Non-compliance');
subplot(2,4,5); imshow(final_image_A); title('Compliance');
subplot(2,4,6); imshow(final_image_B); title('Compliance');
subplot(2,4,7); imshow(final_image_C); title('Non-compliance');
subplot(2,4,8); imshow(final_image_D); title('Non-compliance');
%% Resize the final image
resized_image_A = imresize(final_image_A, [128 64]);
resized_image_B = imresize(final_image_B, [128 64]);
resized_image_C = imresize(final_image_C, [128 64]);
resized_image_D = imresize(final_image_D, [128 64]);

figure, 
subplot(1,4,1); imshow(resized_image_A); title('Compliance');
subplot(1,4,2); imshow(resized_image_B); title('Compliance');
subplot(1,4,3); imshow(resized_image_C); title('Non-compliance');
subplot(1,4,4); imshow(resized_image_D); title('Non-compliance');
%% Grayscale image conversion and LBP feature extraction
gray_image_A = rgb2gray(resized_image_A);
gray_image_B = rgb2gray(resized_image_B);
gray_image_C = rgb2gray(resized_image_C);
gray_image_D = rgb2gray(resized_image_D);

lbp_features_A = extractLBPFeatures(gray_image_A, 'Upright', false);
lbp_features_B = extractLBPFeatures(gray_image_B, 'Upright', false);
lbp_features_C = extractLBPFeatures(gray_image_C, 'Upright', false);
lbp_features_D = extractLBPFeatures(gray_image_D, 'Upright', false);

figure,
subplot(2,4,1); imshow(gray_image_A); title('Compliance');
subplot(2,4,2); imshow(gray_image_B); title('Compliance');
subplot(2,4,3); imshow(gray_image_C); title('Non-compliance');
subplot(2,4,4); imshow(gray_image_D); title('Non-compliance');
subplot(2,4,5); imshow(lbp_features_A); title('Compliance');
subplot(2,4,6); imshow(lbp_features_B); title('Compliance');
subplot(2,4,7); imshow(lbp_features_C); title('Non-compliance');
subplot(2,4,8); imshow(lbp_features_D); title('Non-compliance');
%% Extract HOG features
[featureVector,hogVisualization_A] = extractHOGFeatures(gray_image_A);
[featureVector,hogVisualization_B] = extractHOGFeatures(gray_image_B);
[featureVector,hogVisualization_C] = extractHOGFeatures(gray_image_C);
[featureVector,hogVisualization_D] = extractHOGFeatures(gray_image_D);
figure;
imshow(gray_image_A); 
hold on;
plot(hogVisualization_A);
figure;
imshow(gray_image_B); 
hold on;
plot(hogVisualization_B);
figure;
imshow(gray_image_C); 
hold on;
plot(hogVisualization_C);
figure;
imshow(gray_image_D); 
hold on;
plot(hogVisualization_D);