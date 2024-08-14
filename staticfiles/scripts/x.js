var stats = new Stats();
stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
stats.domElement.style.position = 'absolute';
stats.domElement.style.left = '0px';
stats.domElement.style.top = '50px';
stats.domElement.id = "mystatistics";
//document.body.appendChild( stats.dom );

// Load video
var vidX = 0;
var vidY = 0;
var vid = document.getElementById('vid');
var vidstream;
var devicewidth; // = (window.innerWidth > 0) ? window.innerWidth : screen.width;
var deviceheight;// = (window.innerHeight > 0) ? window.innerHeight : screen.height;

function noScroll(){
  window.scrollTo(0,0);
}

var loader = document.getElementById('loader');
window.addEventListener('scroll',noScroll);

async function loadvideo() {
  if ("mediaDevices" in navigator && "getUserMedia" in navigator.mediaDevices) {
    //rect = document.getElementsByTagName('canvas')[0].getBoundingClientRect();
    vidstream = await navigator.mediaDevices.getUserMedia({
      
      video: {
        width: devicewidth,
        height: deviceheight
      },
      
      audio: false
    });
    vid.srcObject = vidstream;
  }
}

loadvideo();

// Euclidean distance function

function euclid(x, y, mx, my) {
  xsquared = (x - mx) * (x - mx);
  ysquared = (y - my) * (y - my);
  sum = xsquared + ysquared;
  dist = Math.sqrt(sum);
  dist = Math.abs(dist);
  return dist;
}

// Load the model
var model;

async function loadmodel() {
  model = new FaceMesh({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
    }
  });
  model.setOptions({
    selfieMode: true,
    maxNumFaces: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
}

// Graphical User Interface


// THREE.JS Stuff

// video loaded

vid.onloadedmetadata = async function () {
  vidloaded = true;
  devicewidth = window.innerWidth/2;
  deviceheight = vid.videoHeight * (devicewidth / vid.videoWidth);

  var cameradistance = 1000;
  var fieldofview = (Math.atan((deviceheight / 2) / cameradistance));
  fieldofview = (2 * fieldofview * 180 / Math.PI);

  function mapcoords(canvx, canvy) {
    scenex = devicewidth / 2;
    sceney = deviceheight / 2;
    newx = scenex - canvx;
    newy = sceney - canvy;
    return [newx, newy];
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(fieldofview, devicewidth / deviceheight, 0.1, 3000);
  left = mapcoords(0, 0)[0];
  right = mapcoords(devicewidth, 0)[0];
  top = mapcoords(0, 0)[1];
  bottom = mapcoords(0, deviceheight)[1];
  //const camera = new THREE.OrthographicCamera(left, right, top, bottom,1,500);
  const renderer = new THREE.WebGLRenderer({ alpha: false, antialias: true });
  renderer.setSize(devicewidth, deviceheight);

  var canvcontainer = document.getElementById("canvcontainer");
  canvcontainer.appendChild(renderer.domElement);

  // lights
  //scene.add( new THREE.AmbientLight(0xFFFFFF));
  var drlight = new THREE.DirectionalLight('white', 4);
  drlight.castShadows = false;
  drlight.position.set(0, 50, 20);
  scene.add(drlight);

  // Occluding sphere
  /*
  const boxgeometry = new THREE.BoxGeometry( 10, 10, 10 );
  const boxmaterial = new THREE.MeshBasicMaterial( {color: 0x00ff00} );
  const occluder = new THREE.Mesh(boxgeometry,boxmaterial);
  occluder.renderOrder = 1;
  occluder.material.colorWrite=false;
  scene.add(occluder);
  */

  var occluder;
  const omaterial = new THREE.MeshBasicMaterial({ color: 'white' });
  const oader = new THREE.GLTFLoader();
  oader.load(occluderfile, (gltf) => {
    occluder = gltf.scene;
    occluder.traverse((child) => {
      if (child.isMesh) {
        child.material = omaterial;
        child.renderOrder = 1;
        child.material.colorWrite = false;
      }
    });
    occluder.renderOrder = 1;
    scene.add(occluder);
  });

  // Import the glasses 
  var root;
  const goader = new THREE.GLTFLoader();
  goader.load(modelfile,(gltf)=>{
    root = gltf.scene;
    root.renderOrder = 2;
    scene.add(root);
  });

  /*/ pivot */

  /*/ Red square showing face center */
  var plane2geometry= new THREE.PlaneGeometry(100,100);
  var plane2material = new THREE.MeshBasicMaterial({color:'red',depthTest:false});
  var plane2 = new THREE.Mesh(plane2geometry,plane2material);
  plane2.renderOrder = 1;
  plane2.position.x = mapcoords(0,0)[0];
  plane2.position.y = mapcoords(0,0)[1];
  plane2.position.z = 0.1;
  
  scene.add(plane2);
  // Video Plane
  //video texture

  const vidtexture = new THREE.VideoTexture(vid);
  const geometry = new THREE.PlaneGeometry(devicewidth, deviceheight);
  const material = new THREE.MeshPhongMaterial({ depthTest: false });
  material.map = vidtexture;
  const plane = new THREE.Mesh(geometry, material);
  plane.renderOrder = 0;
  scene.add(plane);
  camera.position.z = cameradistance;


  // Distance from virtual camera
  async function animate() {
    // Run the face model
    stats.begin();
    window.removeEventListener('scroll',noScroll);
    await model.send({ image: vid });
    model.onResults((results) => {
      loader.style.display = 'none';
      if (results.multiFaceLandmarks != undefined) {
        for (const keypoints of results.multiFaceLandmarks) {
          rightCoords = keypoints[71];
          leftCoords = keypoints[251];
          bottomCoords = keypoints[4];
          centerCoords = keypoints[168];
          center = keypoints[6];
          topleftx = rightCoords.x * (devicewidth + vidX);
          toplefty = rightCoords.y * (deviceheight + vidY);

          toprightx = leftCoords.x * (devicewidth + vidX);
          toprighty = leftCoords.y * (deviceheight + vidY);

          topcenterx = centerCoords.x * (devicewidth + vidX);
          topcentery = centerCoords.y * (deviceheight + vidY);
          topcenterz = -centerCoords.z

          bottomcenterx = bottomCoords.x * (devicewidth + vidX);
          bottomcentery = bottomCoords.y * (deviceheight + vidY);
          bottomcenterz = -bottomCoords.z
          centerx = mapcoords(center.x * (devicewidth + vidX), 0)[0];
          centery = mapcoords(0, center.y * (deviceheight + vidY))[1];
          centerz = -center.z;

          // plane2.position.x = centerx;
          // plane2.position.y = centery;
          maskWidth = toprightx - topleftx;
          maskHeight = euclid(topcenterx, topcentery, bottomcenterx, bottomcentery);
          left = centerx - parseInt(maskWidth / 2);
          mtop = centery - parseInt(maskHeight / 2);

          // img.style.left = String(left)+"px";
          // img.style.top = String(mtop)+"px";    
          // img.style.left = String(maskX+vidX)+"px";
          // img.style.top = String(maskY+vidY)+"px";

          // Find the angle of rotation along the x-axis
          xadj = keypoints[199].y - keypoints[151].y;
          xopos = -keypoints[151].z - -keypoints[199].z;
          tanx = xopos / xadj;
          rotateXAngle = Math.atan(tanx);

          // Find the angle of rotation along the y-axis    
          yadj = leftCoords.x - rightCoords.x;
          yopos = -leftCoords.z - -rightCoords.z;
          tany = yopos / yadj;
          rotateYAngle = Math.atan(tany);

          // Find angle of rotation along the z-axis
          zadj = rightCoords.x - leftCoords.x;
          zopos = leftCoords.y - rightCoords.y;
          tanz = zopos / zadj;
          rotateZAngle = Math.atan(tanz) / 2;
          
          // Scale the model 
          meshWidth = (yadj * (devicewidth + vidX)) / Math.cos(rotateYAngle);
          //img.style.transform = `rotateZ(${rotateAngle}deg) rotateY(${rotateYAngle}deg)`;

          if (root != undefined) {
            root.position.set(centerx, centery, centerz);
            root.scale.set(meshWidth * modeldata['modelScaleX'], meshWidth * modeldata['modelScaleY'], meshWidth * modeldata['modelScaleZ']);
            root.rotation.set((modeldata['modelRotationX'] * (Math.PI / 180)) + rotateXAngle, (modeldata['modelRotationY'] * (Math.PI / 180)) + rotateYAngle, (modeldata['modelRotationZ'] * (Math.PI / 180)) + rotateZAngle);
          }
          if (occluder != undefined) {
            occluder.position.set(centerx + modeldata['occluderX'], centery + modeldata['occluderY'], centerz + modeldata['occluderZ']);
            occluder.scale.set(meshWidth * modeldata['occluderScaleX'], meshWidth * modeldata['occluderScaleY'], meshWidth * modeldata['occluderScaleZ']);
            occluder.rotation.set((modeldata['modelRotationX'] * (Math.PI / 180)) + rotateXAngle, (modeldata['modelRotationY'] * (Math.PI / 180)) + rotateYAngle, (modeldata['modelRotationZ'] * (Math.PI / 180)) + rotateZAngle);
          }
          // Set the mask's width and height
          //img.style.width = maskWidth.toString() +"px";
          //img.style.height = maskHeight.toString() +"px";

        }
      }
    });
    stats.end();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);

  }
  await loadmodel().then(() => {
    animate();
  });

}