import rospy
from ..main_inference import Test
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class RealtimeInference:
    def __init__(self):
        self.test = Test()
        self.bridge = CvBridge()
        rospy.init_node('RILAB_inference')
        rospy.Subscriber('sync_image_usb', Image, self.image_sub_inference)

    def image_sub_inference(self, image):
        cv2_img = self.bridge.imgmsg_to_cv2(image, "bgr8")
        inference_result = self.test.image_inference(cv2_img)
        self.test.visualization(cv2_img, inference_result)

if __name__ == '__main__':
    realtime_inference = RealtimeInference()
    rospy.spin()
