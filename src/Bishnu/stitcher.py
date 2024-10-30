import cv2
import numpy as np
import glob
import os
from PIL import Image
class PanaromaStitcher:
    def __init__(self):
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.max_size = 2000

    def make_panaroma_for_images_in(self, path):
        image_paths = sorted(glob.glob(os.path.join(path, '*')))
        print(f'Found {len(image_paths)} Images for stitching')

        images = [cv2.imread(img_path) for img_path in image_paths]
        if len(images) < 2:
            print("At least two images are required to create a panorama.")
            return None, []

        homography_matrix_list = []
        stitched_image = images[0]
        center = len(image_paths)//2
        stitched_image = self.read_and_resize_image(image_paths[center])
        print("Image Resizeing..")
        for i in range(center-1,-1,-1 ):
            next_image = self.read_and_resize_image(image_paths[i])
            print("Image Resized \t", i)
            H = self.compute_homography(stitched_image, next_image)
            print("Homography Found",H)
            homography_matrix_list.append(H)
            stitched_image = self.warp_images(stitched_image, next_image, H)
            print("Wrap Completed")
            stitched_image = self.resize_image(stitched_image)
            print("File Stitched")
        set1 = stitched_image

        stitched_image = self.read_and_resize_image(image_paths[center])
        print("Image Resized")
        for i in range(center+1,len(images)):
            next_image = self.read_and_resize_image(image_paths[i])
            print("Image Resized \t", i)
            H = self.compute_homography(stitched_image, next_image)
            print("Homography Found",H)
            homography_matrix_list.append(H)
            stitched_image = self.warp_images(stitched_image, next_image, H)
            print("Wrap Completed")
            stitched_image = self.resize_image(stitched_image)
            print("File Stitched")
        set2 = stitched_image

        stitched_image = set1
        next_image = set2
        H = self.compute_homography(stitched_image, next_image)
        print("Homography Found",H)
        homography_matrix_list.append(H)
        stitched_image = self.warp_images(stitched_image, next_image, H)
        print("Wrap Completed")
        stitched_image = self.resize_image(stitched_image)
        print("File Stitched")
        return stitched_image, homography_matrix_list
    


    def read_and_resize_image(self, image_path):
        img = cv2.imread(image_path)
        return self.resize_image(img)

    def resize_image(self, img):
        h, w = img.shape[:2]
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return img

    # def warp_images(self, img1, img2, H):
    #     h1, w1 = img1.shape[:2]
    #     h2, w2 = img2.shape[:2]

    #     corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    #     corners2 = cv2.perspectiveTransform(corners1, H)

    #     all_corners = np.concatenate((corners2, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)))
    #     [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    #     [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    #     t = [-xmin, -ymin]
    #     Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    #     result = cv2.warpPerspective(img1, Ht.dot(H), (xmax-xmin, ymax-ymin))
    #     result[t[1]:h2+t[1], t[0]:w2+t[0]] = img2

    #     return result

    def warp_images(self, img1, img2, H):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Manual implementation of perspective transform for corners
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]])
        corners2 = self.manual_perspective_transform(corners1, H)

        all_corners = np.concatenate((corners2, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]])))
        [xmin, ymin] = np.int32(all_corners.min(axis=0) - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0) + 0.5)

        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        H_final = Ht.dot(H)

        # Create output image
        result = np.zeros((ymax-ymin, xmax-xmin, 3), dtype=np.uint8)
        
        # Manual warping
        result = self.manual_warp_perspective(img1, result, H_final)
        
        # Copy the second image
        result[t[1]:h2+t[1], t[0]:w2+t[0]] = img2

        return result
    
    def manual_perspective_transform(self, points, H):
        """
        Manual implementation of perspective transform
        """
        transformed_points = []
        for point in points:
            # Convert to homogeneous coordinates
            p = np.array([point[0], point[1], 1])
            # Apply transformation
            p_transformed = np.dot(H, p)
            # Convert back to 2D coordinates
            x = p_transformed[0] / p_transformed[2]
            y = p_transformed[1] / p_transformed[2]
            transformed_points.append([x, y])
        return np.float32(transformed_points)

    def manual_warp_perspective(self, src_img, dst_img, H):
        """
        Manual implementation of perspective warping
        """
        height, width = dst_img.shape[:2]
        
        # Create inverse homography matrix
        H_inv = np.linalg.inv(H)
        
        # For each pixel in the destination image
        for y in range(height):
            for x in range(width):
                # Apply inverse homography
                p = np.array([x, y, 1])
                p_transformed = np.dot(H_inv, p)
                
                # Convert to source image coordinates
                src_x = int(p_transformed[0] / p_transformed[2])
                src_y = int(p_transformed[1] / p_transformed[2])
                
                # Check if the point is within source image bounds
                if 0 <= src_x < src_img.shape[1] and 0 <= src_y < src_img.shape[0]:
                    dst_img[y, x] = src_img[src_y, src_x]
        
        return dst_img

    def bilinear_interpolation(self, img, x, y):
        """
        Perform bilinear interpolation for floating point coordinates
        """
        x1, y1 = int(x), int(y)
        x2, y2 = x1 + 1, y1 + 1
        
        # Get values at corners
        if x2 >= img.shape[1] or y2 >= img.shape[0]:
            return np.zeros(3)
            
        Q11 = img[y1, x1]
        Q21 = img[y1, x2]
        Q12 = img[y2, x1]
        Q22 = img[y2, x2]
        
        # Compute weights
        wx = x - x1
        wy = y - y1
        
        # Interpolate
        top = Q11 * (1 - wx) + Q21 * wx
        bottom = Q12 * (1 - wx) + Q22 * wx
        return top * (1 - wy) + bottom * wy

    
    def compute_homography(self, img1, img2):
        # Detect keypoints and compute descriptors
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)

        # Match descriptors
        matches = self.matcher.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            print("Not enough matches found.")
            return np.eye(3)  # Return identity matrix if not enough matches

        # Get corresponding points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Estimate homography matrix using RANSAC
        H, _ = self.ransac(src_pts, dst_pts, 4, 1000, 5.0)

        return H

    def estimate_homography_dlt(self, src_pts, dst_pts):
        num_points = src_pts.shape[0]
        A = np.zeros((2 * num_points, 9))

        for i in range(num_points):
            x, y = src_pts[i]
            u, v = dst_pts[i]
            A[2*i] = [-x, -y, -1, 0, 0, 0, x*u, y*u, u]
            A[2*i + 1] = [0, 0, 0, -x, -y, -1, x*v, y*v, v]

        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H = H / H[2, 2]  # Normalize

        return H

    def ransac(self, src_pts, dst_pts, min_samples, max_iterations, threshold):
        best_H = None
        best_inliers = 0

        for _ in range(max_iterations):
            # Randomly select minimum number of point correspondences
            idx = np.random.choice(src_pts.shape[0], min_samples, replace=False)
            H = self.estimate_homography_dlt(src_pts[idx], dst_pts[idx])

            # Count inliers
            src_pts_homogeneous = np.column_stack((src_pts, np.ones(src_pts.shape[0])))
            dst_pts_estimated = np.dot(H, src_pts_homogeneous.T).T
            dst_pts_estimated = dst_pts_estimated[:, :2] / dst_pts_estimated[:, 2:]
            errors = np.linalg.norm(dst_pts - dst_pts_estimated, axis=1)
            inliers = np.sum(errors < threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_H = H

        return best_H, best_inliers


