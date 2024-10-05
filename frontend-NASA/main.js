(() => {
  
  const mainApp = async () => {
    const videoContainer = document.querySelector("#video-container")
    videoContainer.src = "http://localhost:8080/api/video"


    const res = await fetch("http://localhost:8080/test");
    const data = await res.json();

    console.log(data);
  }

  mainApp();

})();