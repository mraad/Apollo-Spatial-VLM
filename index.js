import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

// Splat sigma as a fraction of the pixel footprint: neighbours overlap so surfaces look solid.
const SPLAT_SIGMA = 0.6;
// A neighbour whose depth differs by more than this fraction is across an edge, not on the same surface.
const SPLAT_EDGE = 0.12;
// Grazing surfaces stretch splats; past this many pixel footprints fall back to a camera-facing disc.
const SPLAT_MAX_STRETCH = 10;

(function () {
  const form = document.getElementById("demo-form");
  const imageInput = document.getElementById("image");
  const uploadPreview = document.getElementById("upload-preview");
  const depthPreview = document.getElementById("depth-preview");
  const annotatedPreview = document.getElementById("annotated-preview");
  const statusEl = document.getElementById("status");
  const metaEl = document.getElementById("meta");
  const viewerEl = document.getElementById("viewer");
  const submitButton = document.getElementById("submit");
  const promptInput = document.getElementById("prompt");
  const splatsInput = document.getElementById("splats");
  const busyEl = document.getElementById("busy");
  const busyText = document.getElementById("busy-text");
  const busyElapsed = document.getElementById("busy-elapsed");

  let previewUrl = null;
  let pointsObject = null;
  let splatMesh = null;
  let busyTimer = null;
  let markerSpheres = [];
  let cameraGroup = null;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x090b0f);

  const camera = new THREE.PerspectiveCamera(
    60,
    viewerEl.clientWidth / Math.max(viewerEl.clientHeight, 1),
    0.01,
    1000
  );
  camera.position.set(0, 0, 4);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(viewerEl.clientWidth, Math.max(viewerEl.clientHeight, 1));
  viewerEl.appendChild(renderer.domElement);

  const spark = new SparkRenderer({ renderer });
  scene.add(spark);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, -1);

  const grid = new THREE.GridHelper(4, 12, 0x2f3a46, 0x1a222c);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -2;
  scene.add(grid);

  const markerGroup = new THREE.Group();
  scene.add(markerGroup);

  const tooltip = document.createElement("div");
  tooltip.style.cssText =
    "position:fixed;padding:8px 12px;background:rgba(0,0,0,0.85);color:#fff;" +
    "border-radius:6px;font-size:12px;pointer-events:none;display:none;z-index:999;" +
    "font-family:monospace;border:1px solid rgba(255,255,255,0.2);max-width:250px;white-space:pre-line;";
  document.body.appendChild(tooltip);

  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();

  function setStatus(text) {
    statusEl.textContent = text;
  }

  // Overlay on the viewer with a live seconds counter: the server reports no progress, so elapsed time is the honest signal.
  function startBusy(text) {
    const started = performance.now();
    submitButton.disabled = true;
    submitButton.textContent = "Running…";
    busyText.textContent = text;
    busyElapsed.textContent = "0 s";
    busyEl.hidden = false;
    busyTimer = setInterval(() => {
      busyElapsed.textContent = `${Math.floor((performance.now() - started) / 1000)} s`;
    }, 1000);
  }

  function stopBusy() {
    clearInterval(busyTimer);
    busyEl.hidden = true;
    submitButton.disabled = false;
    submitButton.textContent = "Run Demo";
  }

  function renderTargetList(targets) {
    if (!targets.length) {
      metaEl.textContent = "No targets yet.";
      return;
    }
    metaEl.replaceChildren(
      ...targets.map((target) => {
        const coords = `${target.position.x.toFixed(2)}, ${target.position.y.toFixed(2)}, ${target.position.z.toFixed(2)}`;
        const item = document.createElement("div");
        item.className = "target-item";
        const swatch = document.createElement("span");
        swatch.className = "target-swatch";
        swatch.style.background = target.color;
        const text = document.createElement("span");
        text.textContent = `${target.label}  (${coords})`;
        item.append(swatch, text);
        return item;
      })
    );
  }

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  function resize() {
    const width = viewerEl.clientWidth;
    const height = Math.max(viewerEl.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }

  function resetSplats() {
    if (!splatMesh) return;
    scene.remove(splatMesh);
    splatMesh.dispose();
    splatMesh = null;
  }

  function resetPointCloud() {
    resetSplats();
    if (!pointsObject) {
      resetMarkers();
      return;
    }
    scene.remove(pointsObject);
    pointsObject.geometry.dispose();
    pointsObject.material.dispose();
    pointsObject = null;
    resetMarkers();
    resetCameraContext();
  }

  function showSplats() {
    const splats = splatsInput.checked && splatMesh !== null;
    if (splatMesh) splatMesh.visible = splats;
    if (pointsObject) pointsObject.visible = !splats;
  }

  async function decodeRgba(jpegBase64, width, height) {
    const img = new Image();
    img.src = `data:image/jpeg;base64,${jpegBase64}`;
    await img.decode();
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, width, height);
    return ctx.getImageData(0, 0, width, height).data;
  }

  // One flat Gaussian per valid depth pixel, oriented to the local surface (same camera model as demo.py).
  async function loadSplats(depthBase64, rgbBase64, meta) {
    const { width, height } = meta.processed_size;
    const focal = meta.focal_px;
    const depth = decodeFloat32(depthBase64);
    const rgba = await decodeRgba(rgbBase64, width, height);
    const cx = width * 0.5;
    const cy = height * 0.5;

    const unproject = (u, v, d, out) => out.set(((u - cx) * d) / focal, (-(v - cy) * d) / focal, -d);
    // Depth of a neighbour on the same surface, or 0 when it is invalid or across an edge.
    const neighbour = (u, v, d) => {
      if (u < 0 || v < 0 || u >= width || v >= height) return 0;
      const n = depth[v * width + u];
      return n > 0 && Math.abs(n - d) <= SPLAT_EDGE * d ? n : 0;
    };
    // Surface step for one pixel along (du, dv): central difference, else one-sided, else null.
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const tangent = (u, v, d, du, dv, out) => {
      const fwd = neighbour(u + du, v + dv, d);
      const back = neighbour(u - du, v - dv, d);
      if (fwd && back) {
        unproject(u + du, v + dv, fwd, a);
        unproject(u - du, v - dv, back, b);
        return out.subVectors(a, b).multiplyScalar(0.5);
      }
      if (fwd || back) {
        unproject(u, v, d, b);
        unproject(fwd ? u + du : u - du, fwd ? v + dv : v - dv, fwd || back, a);
        return fwd ? out.subVectors(a, b) : out.subVectors(b, a);
      }
      return null;
    };

    const center = new THREE.Vector3();
    const tx = new THREE.Vector3();
    const ty = new THREE.Vector3();
    const normal = new THREE.Vector3();
    const e2 = new THREE.Vector3();
    const scales = new THREE.Vector3();
    const basis = new THREE.Matrix4();
    const quaternion = new THREE.Quaternion();
    const color = new THREE.Color();
    let count = 0;
    for (let i = 0; i < depth.length; i++) if (depth[i] > 0) count++;

    splatMesh = new SplatMesh({
      maxSplats: count,
      constructSplats: (splats) => {
        for (let v = 0; v < height; v++) {
          for (let u = 0; u < width; u++) {
            const i = v * width + u;
            const d = depth[i];
            if (!(d > 0)) continue;
            const footprint = d / focal;
            unproject(u, v, d, center);

            let oriented = tangent(u, v, d, 1, 0, tx) && tangent(u, v, d, 0, 1, ty);
            if (oriented) {
              normal.crossVectors(tx, ty);
              oriented =
                normal.lengthSq() > 0 &&
                tx.length() <= SPLAT_MAX_STRETCH * footprint &&
                ty.length() <= SPLAT_MAX_STRETCH * footprint;
            }
            if (!oriented) {
              // Edge or grazing angle: a camera-facing disc one pixel wide.
              tx.set(footprint, 0, 0);
              ty.set(0, footprint, 0);
              normal.set(0, 0, 1);
            }
            const sx = tx.length() * SPLAT_SIGMA;
            const sy = ty.length() * SPLAT_SIGMA;
            tx.normalize();
            normal.normalize();
            e2.crossVectors(normal, tx);
            basis.makeBasis(tx, e2, normal);
            quaternion.setFromRotationMatrix(basis);
            scales.set(sx, sy, Math.min(sx, sy) * 0.1);
            color.setRGB(rgba[i * 4] / 255, rgba[i * 4 + 1] / 255, rgba[i * 4 + 2] / 255);
            splats.pushSplat(center, scales, quaternion, 1, color);
          }
        }
      },
    });
    await splatMesh.initialized;
    scene.add(splatMesh);
    showSplats();
    return count;
  }

  function createTextSprite(text, color, subtitle) {
    const canvas = document.createElement("canvas");
    const width = 256;
    const height = subtitle ? 100 : 70;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "rgba(0,0,0,0.8)";
    ctx.beginPath();
    ctx.roundRect(4, 4, width - 8, height - 8, 10);
    ctx.fill();

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(4, 4, width - 8, height - 8, 10);
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 26px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text.toUpperCase(), width / 2, subtitle ? 30 : height / 2);

    if (subtitle) {
      ctx.fillStyle = "#aaaaaa";
      ctx.font = "18px sans-serif";
      ctx.fillText(subtitle, width / 2, 65);
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const material = new THREE.SpriteMaterial({
      map: texture,
      depthTest: false,
      sizeAttenuation: true,
    });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(0.35, subtitle ? 0.14 : 0.1, 1);
    return sprite;
  }

  function resetMarkers() {
    markerSpheres = [];
    while (markerGroup.children.length) {
      const child = markerGroup.children[0];
      markerGroup.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (child.material.map) child.material.map.dispose();
        child.material.dispose();
      }
    }
    tooltip.style.display = "none";
  }

  function resetCameraContext() {
    if (!cameraGroup) {
      return;
    }
    while (cameraGroup.children.length) {
      const child = cameraGroup.children[0];
      cameraGroup.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }
    scene.remove(cameraGroup);
    cameraGroup = null;
  }

  function updateCameraContext(markers) {
    resetCameraContext();
    cameraGroup = new THREE.Group();

    const frustumDepth = 0.28;
    const frustumHalfW = 0.18;
    const frustumHalfH = 0.12;
    const frustumPoints = [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(-frustumHalfW, frustumHalfH, -frustumDepth),
      new THREE.Vector3(frustumHalfW, frustumHalfH, -frustumDepth),
      new THREE.Vector3(frustumHalfW, -frustumHalfH, -frustumDepth),
      new THREE.Vector3(-frustumHalfW, -frustumHalfH, -frustumDepth),
    ];
    const frustumIndices = [
      [0, 1], [0, 2], [0, 3], [0, 4],
      [1, 2], [2, 3], [3, 4], [4, 1],
    ];
    for (const [a, b] of frustumIndices) {
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([frustumPoints[a], frustumPoints[b]]),
        new THREE.LineBasicMaterial({ color: 0x7cc7ff, transparent: true, opacity: 0.95 })
      );
      cameraGroup.add(line);
    }

    const radius = pointsObject?.geometry?.boundingSphere?.radius || 1;
    const defaultArrowLength = Math.max(0.1, radius * 0.06);
    for (const marker of markers) {
      const color = marker.color;
      const target = new THREE.Vector3(
        marker.position.x,
        marker.position.y,
        marker.position.z
      );
      const distance = target.length();
      if (distance < 1e-8) continue;
      const arrowLength = Math.min(defaultArrowLength, distance);
      const dir = target.clone().normalize();

      const ray = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(0, 0, 0),
          target.clone(),
        ]),
        new THREE.LineDashedMaterial({
          color,
          transparent: true,
          opacity: 0.45,
          dashSize: 0.035,
          gapSize: 0.03,
        })
      );
      ray.computeLineDistances();
      cameraGroup.add(ray);

      const headLength = Math.min(0.06, arrowLength * 0.32);
      const shaftLength = Math.max(arrowLength - headLength, 0.02);
      const shaftRadius = 0.018;
      const headRadius = 0.045;
      const material = new THREE.MeshBasicMaterial({ color });

      const shaft = new THREE.Mesh(
        new THREE.CylinderGeometry(shaftRadius, shaftRadius, shaftLength, 12),
        material
      );
      shaft.position.copy(dir.clone().multiplyScalar(shaftLength * 0.5));
      shaft.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir
      );
      cameraGroup.add(shaft);

      const cone = new THREE.Mesh(
        new THREE.ConeGeometry(headRadius, headLength, 12),
        material.clone()
      );
      cone.position.copy(dir.clone().multiplyScalar(shaftLength + headLength * 0.5));
      cone.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir
      );
      cameraGroup.add(cone);
    }

    scene.add(cameraGroup);
  }

  function updateMarkers(markers) {
    resetMarkers();

    for (const marker of markers) {
      const { x, y, z } = marker.position;
      const color = marker.color;
      const coords = `(${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)})`;

      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.04),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.92 })
      );
      sphere.position.set(x, y, z);
      sphere.userData = { marker, coords };
      markerGroup.add(sphere);
      markerSpheres.push(sphere);

      const stem = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(x, y, z),
          new THREE.Vector3(x, y + 0.18, z),
        ]),
        new THREE.LineBasicMaterial({ color })
      );
      markerGroup.add(stem);

      const sprite = createTextSprite(
        marker.label,
        color,
        `${coords} ${(marker.confidence * 100).toFixed(0)}%`
      );
      sprite.position.set(x, y + 0.28, z);
      markerGroup.add(sprite);
    }
  }

  function decodeFloat32(base64) {
    return new Float32Array(Uint8Array.from(atob(base64), (c) => c.charCodeAt(0)).buffer);
  }

  function loadPointCloud(points, colors) {
    resetPointCloud();

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(decodeFloat32(points), 3)
    );
    geometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(decodeFloat32(colors), 3)
    );

    const material = new THREE.PointsMaterial({
      size: 0.025,
      vertexColors: true,
      sizeAttenuation: true,
    });

    pointsObject = new THREE.Points(geometry, material);
    scene.add(pointsObject);

    geometry.computeBoundingSphere();
    const sphere = geometry.boundingSphere;
    if (!sphere) {
      return;
    }

    controls.target.copy(sphere.center);
    const radius = Math.max(sphere.radius, 0.25);
    camera.position.set(
      sphere.center.x,
      sphere.center.y,
      sphere.center.z + radius * 2.4
    );
    camera.near = Math.max(radius / 500, 0.01);
    camera.far = Math.max(radius * 20, 100);
    camera.updateProjectionMatrix();
    controls.update();
  }

  imageInput.addEventListener("change", () => {
    const file = imageInput.files && imageInput.files[0];
    if (!file) {
      return;
    }
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    previewUrl = URL.createObjectURL(file);
    uploadPreview.src = previewUrl;
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = imageInput.files && imageInput.files[0];
    if (!file) {
      setStatus("Choose an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);
    const promptValue = promptInput.value.trim();
    formData.append("prompt", promptValue);

    startBusy(promptValue ? "Estimating depth and finding targets" : "Estimating depth");
    setStatus("Uploading image and running Depth Anything 3...");
    metaEl.textContent = "Running...";

    try {
      const response = await fetch("/api/infer", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || "Inference failed.");
      }

      depthPreview.src = `data:image/png;base64,${result.depth_preview}`;
      if (result.annotated_preview) {
        annotatedPreview.src = `data:image/png;base64,${result.annotated_preview}`;
      } else {
        annotatedPreview.removeAttribute("src");
      }
      const targets = result.targets_3d || [];
      renderTargetList(targets);
      loadPointCloud(result.points, result.colors);
      let splatNote;
      try {
        const started = performance.now();
        const splatCount = await loadSplats(result.depth, result.rgb, result.meta);
        splatNote = `${splatCount} splats (built in ${Math.round(performance.now() - started)} ms)`;
      } catch (error) {
        // Splats are optional: keep the point cloud and markers when they fail.
        resetSplats();
        showSplats();
        splatNote = `no splats (${error.message || error})`;
      }
      updateMarkers(targets);
      updateCameraContext(targets);
      setStatus(
        `Rendered ${result.meta.point_count} points, ${splatNote} and ${result.meta.target_count || 0} prompt targets. ` +
          `Field of view ${Math.round(result.meta.fov_deg)}° (${result.meta.fov_source === "exif" ? "from EXIF" : "assumed"}).`
      );
    } catch (error) {
      setStatus(String(error));
      metaEl.textContent = "No targets yet.";
      annotatedPreview.removeAttribute("src");
      resetPointCloud();
    } finally {
      stopBusy();
    }
  });

  renderer.domElement.addEventListener("mousemove", (event) => {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(markerSpheres);

    if (!hits.length) {
      tooltip.style.display = "none";
      renderer.domElement.style.cursor = "default";
      return;
    }

    const { marker, coords } = hits[0].object.userData;
    const title = document.createElement("b");
    title.textContent = marker.label.toUpperCase();
    tooltip.replaceChildren(
      title,
      `\nPosition: ${coords}\nConfidence: ${(marker.confidence * 100).toFixed(0)}%\nPixel: (${marker.pixel.x}, ${marker.pixel.y})`
    );
    tooltip.style.display = "block";
    tooltip.style.left = `${event.clientX + 14}px`;
    tooltip.style.top = `${event.clientY + 14}px`;
    renderer.domElement.style.cursor = "pointer";
  });

  splatsInput.addEventListener("change", showSplats);
  window.addEventListener("resize", resize);
  resize();
  animate();
})();
