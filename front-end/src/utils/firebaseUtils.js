import { getStorage, ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { auth, db } from '../firebase';
import { doc, updateDoc } from "firebase/firestore";

export const uploadProfilePicture = async (userId, file) => {
  if (!file) return null;

  const storage = getStorage();
  const storageRef = ref(storage, `profilePictures/${userId}`);

  try {
    const snapshot = await uploadBytes(storageRef, file);
    const downloadURL = await getDownloadURL(snapshot.ref);
    const userRef = doc(db, "users", userId);
    await updateDoc(userRef, {
      profilePicture: downloadURL
    });

    return downloadURL;
  } catch (error) {
    console.error("Error uploading profile picture:", error);
    throw error;
  }
};