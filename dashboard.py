"""
ReID Dashboard — run with: streamlit run dashboard.py
"""
import io
import time
import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="ReID Dashboard",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")

    server_url = st.text_input("Server URL", value="http://localhost:5005")

    # Connection indicator
    try:
        stats = requests.get(f"{server_url}/stats", timeout=2).json()
        st.success("● Server online")
    except Exception:
        stats = None
        st.error("● Server offline")

    st.divider()

    # Threshold controls
    st.subheader("Match Thresholds")
    current_p = THRESHOLDS_CACHE = 0.6
    current_v = 0.5
    person_thresh  = st.slider("Person",  0.1, 1.0, current_p, 0.05)
    vehicle_thresh = st.slider("Vehicle", 0.1, 1.0, current_v, 0.05)
    if st.button("Apply", use_container_width=True):
        try:
            requests.patch(
                f"{server_url}/config",
                json={"person": person_thresh, "vehicle": vehicle_thresh},
                timeout=2,
            )
            st.success("Thresholds updated")
        except Exception:
            st.error("Failed to reach server")

    st.divider()

    # Gallery reset
    st.subheader("Gallery")
    if st.button("🗑️ Reset Gallery", type="secondary", use_container_width=True):
        try:
            requests.post(f"{server_url}/gallery/reset", timeout=3)
            st.cache_data.clear()
            st.success("Gallery cleared")
            time.sleep(0.5)
            st.rerun()
        except Exception:
            st.error("Failed")

    # CSV download
    try:
        csv_bytes = requests.get(f"{server_url}/get_log", timeout=5).content
        st.download_button(
            "⬇️ Download CSV",
            csv_bytes,
            "reid_log.csv",
            "text/csv",
            use_container_width=True,
        )
    except Exception:
        pass

    st.divider()
    refresh_rate = st.slider("Auto-refresh (s)", 1, 15, 3)


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎥 Multi-Camera ReID Dashboard")

if not stats:
    st.warning("Cannot reach the ReID server. Make sure `server.py` is running.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("🚶 Persons",    stats["identities"].get("person",  0))
c2.metric("🚗 Vehicles",   stats["identities"].get("vehicle", 0))
c3.metric("👁️ Sightings",  stats["total_sightings"])
c4.metric("📷 Cameras",    stats["active_cameras"])

st.divider()


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_crop(server_url: str, global_id: str):
    try:
        r = requests.get(f"{server_url}/identities/{global_id}/crop", timeout=3)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None


def identity_grid(identities: list, server_url: str):
    if not identities:
        st.info("None detected yet.")
        return

    cols = st.columns(5)
    for i, ident in enumerate(identities):
        gid = ident["global_id"]
        with cols[i % 5]:
            crop_bytes = fetch_crop(server_url, gid)
            if crop_bytes:
                st.image(Image.open(io.BytesIO(crop_bytes)), use_container_width=True)
            else:
                st.markdown(
                    "<div style='height:120px;background:#222;border-radius:6px;"
                    "display:flex;align-items:center;justify-content:center;"
                    "color:#666;font-size:28px'>?</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(f"**{gid}**")
            st.caption(f"Seen {ident['sightings']}×  ·  {ident['last_seen'][5:16]}")

            with st.expander("Sighting history"):
                try:
                    r = requests.get(
                        f"{server_url}/identities/{gid}/sightings?limit=10",
                        timeout=2,
                    )
                    sightings = r.json().get("sightings", [])
                    if sightings:
                        for s in sightings:
                            new_badge = " 🆕" if s["is_new"] else ""
                            st.text(
                                f"{s['timestamp'][5:19]}  [{s['camera_id']}]"
                                f"  score={s['score']:.2f}{new_badge}"
                            )
                    else:
                        st.text("No sightings recorded.")
                except Exception:
                    st.text("Unavailable")


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=2)
def fetch_latest_frame(server_url: str, camera_id: str):
    try:
        r = requests.get(f"{server_url}/camera/{camera_id}/frame", timeout=2)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None


def livestream_grid(server_url: str):
    try:
        cams = requests.get(f"{server_url}/cameras", timeout=2).json().get("cameras", [])
    except Exception:
        cams = []

    if not cams:
        st.info("No active camera streams yet. Start the sim cameras or edge clients.")
        return

    cols = st.columns(2)
    for i, cam in enumerate(cams):
        with cols[i % 2]:
            st.subheader(cam)
            frame_bytes = fetch_latest_frame(server_url, cam)
            if frame_bytes:
                st.image(Image.open(io.BytesIO(frame_bytes)), use_container_width=True)
            else:
                st.info("Waiting for frame...")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_vehicles, tab_persons, tab_feed, tab_stream = st.tabs(
    ["🚗 Vehicles", "🚶 Persons", "📋 Live Feed", "📺 Livestream"]
)

with tab_vehicles:
    try:
        vehicles = requests.get(
            f"{server_url}/identities?category=vehicle", timeout=2
        ).json().get("identities", [])
    except Exception:
        vehicles = []
    identity_grid(vehicles, server_url)

with tab_persons:
    try:
        persons = requests.get(
            f"{server_url}/identities?category=person", timeout=2
        ).json().get("identities", [])
    except Exception:
        persons = []
    identity_grid(persons, server_url)

with tab_feed:
    try:
        logs = requests.get(
            f"{server_url}/get_recent_logs?lines=50", timeout=2
        ).json().get("logs", [])
    except Exception:
        logs = []

    if logs:
        for line in reversed(logs[-30:]):
            parts = line.split(",")
            if len(parts) >= 5:
                ts, cam, gid, cat, score = parts[0], parts[1], parts[2], parts[3], parts[4]
                icon = "🚶" if cat == "person" else "🚗"
                st.markdown(
                    f"`{ts[5:19]}` &nbsp; **{cam}** &nbsp; {icon} `{gid}` &nbsp; "
                    f"score `{float(score):.3f}`"
                )
    else:
        st.info("No sightings yet — start an edge client to begin.")

with tab_stream:
    livestream_grid(server_url)

# ── Auto-refresh ───────────────────────────────────────────────────────────────
time.sleep(refresh_rate)
st.rerun()
