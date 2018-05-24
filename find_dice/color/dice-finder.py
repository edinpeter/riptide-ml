import cv2
import sys
import numpy as np
import math

colors = [(0,0,255),
          (0,255,0),
          (255,0,255),
          (255,255,0),
          (0,255,255),
          (127,0,127)]

area = lambda tlp, brp : ((brp[0] - tlp[0]) * (brp[1] - tlp[1]))

CLEAN_CLUSTERS = True
CLEAN_CLUSTERS_DISTANCE = 40
MERGE_CLOSE_CLUSTERS = True
MERGE_CLOSE_DISTANCE = 40

def distance(pt1, pt2):
        x = pt1[0] - pt2[0]
        y = pt1[1] - pt2[1]
        return math.sqrt(x**2 + y**2)

def get_centers(contours):
    centers = list()
    for cnt in contours:
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        centers.append((cx,cy))
    return centers

def get_clusters(centers):
    ## Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    centers = np.float32(centers)
    flags = cv2.KMEANS_RANDOM_CENTERS
    min_compactness = sys.maxsize
    min_centers, min_labels = None, None

    for classes in range(1, min(len(centers) + 1, 7)):
        compactness,labels,cluster_centers = cv2.kmeans(centers, classes, None, criteria, 10, flags)
        if compactness < min_compactness:
            min_centers, min_labels = cluster_centers, labels
    return min_centers, min_labels

def circumference_check(contour):
    area = cv2.contourArea(contour)
    equi_diameter = np.sqrt(4*area/np.pi)
    perimeter = cv2.arcLength(contour,True)
    ratio = equi_diameter * math.pi / perimeter
    return ratio > .9

def area_check(contour, size_limit=200, size_min = 20):
    area = cv2.contourArea(contour)
    return area < size_limit and area > size_min

def get_countours(image, black=40):
    lower_black = (0,0,0)
    upper_black = (black,black,black)
    mask = cv2.inRange(image, lower_black, upper_black)
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    reduced_contours = list()
    for cnt in contours:
    	if area_check(cnt) and circumference_check(cnt):
			reduced_contours.append(cnt)

    return mask, image, reduced_contours

def draw_clusters(image, labels, centers):
    for label, center in zip(labels, centers):
        try:
            cv2.circle(image, center, 4, colors[label[0]], -1)
        except:
            pass
    return image

def get_cluster_lists(labels, centers, image=None):
    lists = dict()
    for label, center in zip(labels, centers):
            if label[0] not in lists:
                    lists[label[0]] = list()
            lists[label[0]].append(center)
	
    lists = clean_clusters(lists) if CLEAN_CLUSTERS else lists
    if image is not None:
            for label in lists.keys():
                    for center in lists[label]:
                            for center2 in lists[label]:
                                    cv2.line(image, center, center2, colors[label])
    
    return lists


def largest_distance(list):
        max_distance = -1
        for point in list:
                for point2 in list:
                        dist = distance(point, point2)
                        max_distance = max(max_distance, dist)
        return max_distance

def dice_subimages(cluster_lists, cluster_centers, image):
        for label in cluster_lists.keys():
                dim = largest_distance(cluster_lists[label])
                dim = 50 if dim == 0 else dim
                center = new_cluster_center(cluster_lists[label])
                #tlp = (int(center[0] - dim), int(center[1] - dim))
                #brp = (int(center[0] + dim), int(center[1] + dim))
                tlp_x = int(center[0] - dim)
                tlp_y = int(center[1] - dim)
                brp_x = int(center[0] + dim)
                brp_y = int(center[1] + dim)

                tlp = (tlp_x, tlp_y)
                brp = (brp_x, brp_y)

                if area(tlp, brp) > 100:
                        factor = 3
                        for i in range(-5, 10):
                                tlp_x -= factor
                                tlp_y -= factor
                                brp_x += factor
                                brp_y += factor
                                tlp = (tlp_x, tlp_y)
                                brp = (brp_x, brp_y)
                                cv2.rectangle(image, tlp, brp, colors[label])
        return image

def new_cluster_center(points):
	l = len(points)
	x = sum(zip(*points)[0])
	y = sum(zip(*points)[1])
	return (x / float(l), y / float(l))

def replace_labels(old, new, labels):
    for label in labels:
            label[0] = new if label[0] == old else label[0]
    return labels

def merge_close_clusters(ccenters, labels):
    for i in range(0, len(ccenters)):
            for j in range(0, len(ccenters)):
                    if distance(ccenters[i], ccenters[j]) < MERGE_CLOSE_DISTANCE:
                            labels = replace_labels(i, j, labels)
    return labels

def clean_clusters(cluster_lists):
	for label in cluster_lists.keys():
		cluster_dists = list()
		for point1 in cluster_lists[label]:
			point1_sum = 0
			for point2 in cluster_lists[label]:
				point1_sum += 1 if distance(point1, point2) > CLEAN_CLUSTERS_DISTANCE else 0
			cluster_dists.append(point1_sum)
		to_del = list()
		for i in range(len(cluster_dists)):
			if cluster_dists[i] > len(cluster_lists[label]) / 2:
				to_del.append(cluster_lists[label][i])
		for delete in to_del:
			cluster_lists[label].remove(delete)
	for label in cluster_lists.keys():
		if len(cluster_lists[label]) == 0:
			del cluster_lists[label]

	return cluster_lists              

if __name__ == "__main__":
    image = cv2.imread('samples/85.png', cv2.IMREAD_UNCHANGED)
    print image.shape
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    except:
        pass
    image = cv2.blur(image, (7, 7))
    mask, image, contours = get_countours(image, black=100)

    cv2.imshow('image,', image)
    cv2.waitKey(0)
    cv2.imshow('mask,', mask)
    cv2.waitKey(0)

    centers = get_centers(contours)
    min_clusters, min_labels = get_clusters(centers)
    if min_clusters is not None:
            min_labels = merge_close_clusters(min_clusters, min_labels) if MERGE_CLOSE_CLUSTERS else min_labels

            cluster_lists = get_cluster_lists(min_labels, centers, image)
            #cluster_lists = clean_clusters(cluster_lists)

            image = draw_clusters(image, min_labels, centers)
            image = dice_subimages(cluster_lists, min_clusters, image)

            cv2.imshow('clusters', image)
            cv2.waitKey(0)

