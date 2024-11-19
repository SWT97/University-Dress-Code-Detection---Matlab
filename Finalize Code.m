% Datasets Paths
compliance_folder = "\OK";
non_compliance_folder = "\Not OK";

% List of image files in each folder
compliance_images = dir(fullfile(compliance_folder, '*.jpg')); 
non_compliance_images = dir(fullfile(non_compliance_folder, '*.jpg'));

% Empty arrays for feature vectors and labels
X = [];
y = [];

% Preprocess the image
function preprocessed_img = preprocess_image(image) 
    % RGB image convert to YCbCr image
    ycbcr_image = rgb2ycbcr(image);

    % Extraction of Y channel
    Y = ycbcr_image(:,:,1);

    % Apply CLAHE to the Y channel
    clahe_Y = adapthisteq(Y);

    % Place back modified Y channel to the YCbCr image
    ycbcr_image(:,:,1) = clahe_Y;

    % Assign preprocessed image as YCbCr image
    preprocessed_img = ycbcr_image;
end

% Image segmentation for skin and clothing
function [skin_mask, clothing_mask, combined_mask, skin_segmented, clothing_segmented, final_image] = segment_image(preprocessed_img, image)
    ycbcr_img = preprocessed_img;  % Use the preprocessed image

    % Define thresholds for skin in YCbCr color space
    lower_skin = [0, 77, 133];  
    upper_skin = [150, 127, 173];

    % Detect skin regions based on the YCbCr thresholds
    skin_mask = (ycbcr_img(:,:,2) >= lower_skin(2)) & (ycbcr_img(:,:,2) <= upper_skin(2)) & ...
                (ycbcr_img(:,:,3) >= lower_skin(3)) & (ycbcr_img(:,:,3) <= upper_skin(3));

    % Perform morphological operations to clean the skin mask
    skin_mask = imopen(skin_mask, strel('disk', 6));

    % Segment skin regions
    skin_segmented = image;
    skin_segmented(repmat(~skin_mask, [1 1 3])) = 0;  

    % Convert to HSV for clothing segmentation
    hsv_image = rgb2hsv(image);
    
    % Define thresholds for clothing in HSV color space
    lower_cloth = [0, 0, 0];
    upper_cloth = [1, 1, 0.4];
    
    % Detect clothing regions based on HSV thresholds
    clothing_mask = (hsv_image(:,:,1) >= lower_cloth(1)) & (hsv_image(:,:,1) <= upper_cloth(1)) & ...
                    (hsv_image(:,:,2) >= lower_cloth(2)) & (hsv_image(:,:,2) <= upper_cloth(2)) & ...
                    (hsv_image(:,:,3) >= lower_cloth(3)) & (hsv_image(:,:,3) <= upper_cloth(3));

    % Remove skin areas from the clothing mask
    clothing_mask = clothing_mask & ~skin_mask;

    % Clean the clothing mask using morphological operations
    clothing_mask = imclose(clothing_mask, strel('disk', 6));

    % Segment clothing regions
    clothing_segmented = image;
    clothing_segmented(repmat(~clothing_mask, [1 1 3])) = 0;

    % Combine skin and clothing masks
    combined_mask = ~skin_mask | clothing_mask;

    % Set all non-skin and non-clothing areas to white
    final_image = image;
    final_image(repmat(~combined_mask, [1 1 3])) = 255;
end

% Feature extraction using LBP and HOG
function features = extract_features(final_image)
    % Resize the masked image for feature extraction
    resized_image = imresize(final_image, [128 64]); 

    % Convert the resized image to grayscale
    gray_image = rgb2gray(resized_image);
    
    % Extract LBP features
    lbp_features = extractLBPFeatures(gray_image, 'Upright', false);
    
    % Extract HOG features
    hog_features = extractHOGFeatures(gray_image);

    % Combine HOG and LBP features
    features = [hog_features, lbp_features];
end

% Labelling Process: Process compliance images (Label = 1)
for i = 1:numel(compliance_images)
    img_path = fullfile(compliance_folder, compliance_images(i).name);
    img = imread(img_path);
    
    % Preprocess, segmentation, and feature extraction
    preprocessed_img = preprocess_image(img);
    [~, ~, ~, ~, ~, final_image] = segment_image(preprocessed_img, img);
    
    % Extract features
    features = extract_features(final_image);
    
    % Store features and labels
    X = [X; features(:)'];  % Reshape features to row vector before appending
    y = [y; 1];  % Label = 1 for compliance
end

% Labelling Process: Process non-compliance images (Label = 0)
for i = 1:numel(non_compliance_images)
    img_path = fullfile(non_compliance_folder, non_compliance_images(i).name);
    img = imread(img_path);
    
    % Preprocess, segmentation, and feature extraction
    preprocessed_img = preprocess_image(img);
    [~, ~, ~, ~, ~, final_image] = segment_image(preprocessed_img, img);
    
    % Extract features
    features = extract_features(final_image);
    
    % Store features and labels
    X = [X; features(:)'];  % Reshape features to row vector before appending
    y = [y; 0];  % Label = 0 for non-compliance
end

% Display size of the feature matrix and labels
disp(['Number of images processed: ', num2str(size(X, 1))]);
disp(['Number of labels: ', num2str(length(y))]);

% Validation for sizes match
if size(X, 1) ~= length(y)
    error('Mismatch between number of features and labels!');
end

% Convert X and y to double
X = double(X);
y = double(y);

% Set random seed as 42
rng(42);

% Split the data into training and testing sets (80% training, 20% testing)
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :); 

% Declare SVMModel as global
global SVMModel;

% Train the SVM model on the training set
SVMModel = fitcsvm(X_train, y_train, 'KernelFunction', 'linear', 'Standardize', true);

% Make predictions on the test set
[predicted_labels, scores] = predict(SVMModel, X_test);

% Evaluate accuracy and display it
accuracy = sum(predicted_labels == y_test) / numel(y_test);
disp(['Test Set Accuracy: ', num2str(accuracy)]);

% GUI 
function dresscode_detection_gui
    hFig = uifigure('Name', 'Dress Code Detection', 'Position', [100 100 600 400]);
    
    % Buttons
    hLoadButton = uibutton(hFig, 'Text', 'Load Images', 'Position', [20 350 100 30], 'ButtonPushedFcn', @(btn,event) loadImagesCallback());
    hTestButton = uibutton(hFig, 'Text', 'Test', 'Position', [130 350 100 30], 'ButtonPushedFcn', @(btn,event) testImageCallback());
    hNextButton = uibutton(hFig, 'Text', 'Next', 'Position', [240 350 100 30], 'ButtonPushedFcn', @(btn,event) nextImageCallback());
    hPrevButton = uibutton(hFig, 'Text', 'Previous', 'Position', [350 350 100 30], 'ButtonPushedFcn', @(btn,event) prevImageCallback());

    % Axes to display the image
    hAxes = uiaxes(hFig, 'Position', [50 50 400 300]);
    title(hAxes, 'Loaded Image');
    
    % Text area for compliance/non-compliance
    hResult = uitextarea(hFig, 'Position', [450 150 80 35], 'Editable', 'off');
    
    % Global variables to store image data
    global imageFiles currentImageIndex;
    imageFiles = {};
    currentImageIndex = 0;
    
    % Load images callback
    function loadImagesCallback()
        folderName = uigetdir(); 
        if folderName
            imageFiles = dir(fullfile(folderName, '*.jpg'));
            if ~isempty(imageFiles)
                currentImageIndex = 1;
                displayImage(currentImageIndex);
            else
                uialert(hFig, 'No images found in the selected folder', 'Error');
            end
        end
    end

    % Test image callback
    function testImageCallback()
        global SVMModel;  % Access the global SVMModel variable
        if currentImageIndex > 0 && currentImageIndex <= numel(imageFiles)
            img_path = fullfile(imageFiles(currentImageIndex).folder, imageFiles(currentImageIndex).name);
            img = imread(img_path);
            preprocessed_img = preprocess_image(img);
            [~, ~, ~, ~, ~, final_image] = segment_image(preprocessed_img, img);
            features = extract_features(final_image);
            label = predict(SVMModel, features);  % Predict label using trained model
            
            % Display result
            if label == 1
                hResult.Value = 'Compliance';
            else
                hResult.Value = 'Non-compliance';
            end
        end
    end

    % Display the current image in the axes
    function displayImage(index)
        img_path = fullfile(imageFiles(index).folder, imageFiles(index).name);
        img = imread(img_path);
        imshow(img, 'Parent', hAxes);
    end

    % Next image callback
    function nextImageCallback()
        if currentImageIndex < numel(imageFiles)
            currentImageIndex = currentImageIndex + 1;
            displayImage(currentImageIndex);
            hResult.Value = '';  % Clear the text area when changing images
        else
            uialert(hFig, 'This is the last image.', 'Info');
        end
    end

    % Previous image callback
    function prevImageCallback()
        if currentImageIndex > 1
            currentImageIndex = currentImageIndex - 1;
            displayImage(currentImageIndex);
            hResult.Value = '';  % Clear the text area when changing images
        else
            uialert(hFig, 'This is the first image.', 'Info');
        end
    end
end

% Save the model for GUI
save('SVMModel.mat', 'SVMModel');  

dresscode_detection_gui