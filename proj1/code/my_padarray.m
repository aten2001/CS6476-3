function padded_image = my_padarray(image, pad_size, pad_type)
% This function is intended to behave like the built in function padarray()
% but at the moment only supports symmetric image padding and 2D padding sizes.

    image_size = size(image);
    if size(image_size)== [1 2];
        % grayscale image
        image_size = [image_size 1];
    end
    padded_image_size = image_size + [2*pad_size 0];
    padded_image = zeros(padded_image_size);
    if (~strcmp(pad_type,'symmetric'))
        return
    end
    for k=1:image_size(3)
        for j=1:image_size(2)
            for i=1:pad_size(1)
                % top-center padding
                padded_image(i, pad_size(2)+j, k) = image(pad_size(1)-i+1, j,k);
                % bottom-center padding
                padded_image(pad_size(1)+image_size(1)+i, pad_size(2)+j, k) = image(image_size(1)-i+1, j,k);
            end
        end
        for i=1:image_size(1)
            for j=1:pad_size(2)
                % left-center padding
                padded_image(pad_size(1)+i, j, k) = image(i, pad_size(2)-j+1, k);
                % right-center padding
                padded_image(pad_size(1)+i, pad_size(2)+image_size(2)+j, k) = image(i, image_size(2)-j+1, k);
            end
        end
        % central image
        for j=1:image_size(2)
            for i=1:image_size(1)
                padded_image(i+pad_size(1), j+pad_size(2), k) = image(i, j, k);
            end
        end
        
        for i=1:pad_size(1)
            for j=1:pad_size(2)
                % top left corner
                padded_image(i, j, k) = padded_image(i, 2*pad_size(2)-j+1, k);
                % top right corner
                padded_image(i, padded_image_size(2)-j+1, k) = padded_image(i, padded_image_size(2)-2*pad_size(2)+j, k); 
                % bottom left corner
                padded_image(padded_image_size(1)-pad_size(1)+i, j, k) = padded_image(padded_image_size(1)-pad_size(1)+i, 2*pad_size(2)-j+1, k);
                % bottom right corner
                padded_image(padded_image_size(1)-pad_size(1)+i, padded_image_size(2)-j+1, k) = padded_image(padded_image_size(1)-pad_size(1)+i, padded_image_size(2)-2*pad_size(2)+j, k); 
            end
        end
    end