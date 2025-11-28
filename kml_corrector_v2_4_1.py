"""
KML Corrector v2.4.1 - COORDINATE SYSTEM FOR SHAPEFILE ONLY
Usage: streamlit run kml_corrector_v2_4_1.py

IMPORTANT:
- KML files: Always exported in ORIGINAL coordinate system (unchanged)
- Shapefile: Can be converted to target coordinate system
- This preserves KML compatibility while allowing shapefile conversions
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
from lxml import etree
import numpy as np
from typing import List, Tuple, Dict, Optional
import zipfile
from io import BytesIO

try:
    import shapefile
    SHAPEFILE_AVAILABLE = True
except ImportError:
    SHAPEFILE_AVAILABLE = False
    shapefile = None

try:
    from pyproj import Transformer, CRS
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    Transformer = None
    CRS = None

st.set_page_config(page_title="KML Corrector v2.4.1", layout="wide")

# Common coordinate systems
CRS_OPTIONS = {
    'WGS84 (Lat/Lon)': 'EPSG:4326',
    'UTM Zone 43N': 'EPSG:32643',
    'UTM Zone 43S': 'EPSG:32743',
    'UTM Zone 42N': 'EPSG:32642',
    'UTM Zone 42S': 'EPSG:32742',
    'UTM Zone 44N': 'EPSG:32644',
    'UTM Zone 44S': 'EPSG:32744',
    'UTM Zone 41N': 'EPSG:32641',
    'UTM Zone 41S': 'EPSG:32741',
    'UTM Zone 45N': 'EPSG:32645',
    'UTM Zone 45S': 'EPSG:32745',
    'Web Mercator': 'EPSG:3857',
    'India State Plane': 'EPSG:7755',
}

class KMLCorrector:
    """Enhanced KML corrector with coordinate system support"""
    
    NS = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    def __init__(self, kml_content: str, filename: str):
        """Initialize with KML content"""
        self.filename = Path(filename).stem
        self.original_content = kml_content
        try:
            self.root = etree.fromstring(kml_content.encode('utf-8'))
        except Exception as e:
            raise ValueError(f"Invalid KML file: {str(e)}")
    
    def is_likely_latitude(self, value: float) -> bool:
        """Check if value is likely latitude (-90 to 90)"""
        return -90 <= value <= 90
    
    def is_likely_longitude(self, value: float) -> bool:
        """Check if value is likely longitude (-180 to 180)"""
        return -180 <= value <= 180
    
    def auto_detect_coordinate_order(self, val1: float, val2: float) -> Tuple[float, float]:
        """Auto-detect if coordinates are (lon, lat) or (lat, lon)"""
        if self.is_likely_longitude(val1) and self.is_likely_latitude(val2):
            if abs(val1) <= 90 and abs(val2) > 90:
                return val2, val1
            return val1, val2
        
        if self.is_likely_latitude(val1) and self.is_likely_longitude(val2):
            if abs(val1) > 90:
                return val1, val2
            if abs(val1) <= 90 and abs(val2) > 90:
                return val2, val1
            return val2, val1
        
        if abs(val1) > 180:
            return val2, val1
        
        if abs(val2) > 180:
            return val2, val1
        
        return val1, val2
    
    def extract_coordinates(self, coord_string: str) -> List[Tuple[float, float, float]]:
        """Parse KML coordinates with auto-detection of order"""
        coords = []
        for coord in coord_string.strip().split():
            if coord:
                parts = coord.split(',')
                if len(parts) >= 2:
                    try:
                        val1 = float(parts[0])
                        val2 = float(parts[1])
                        alt = float(parts[2]) if len(parts) > 2 else 0.0
                        
                        lon, lat = self.auto_detect_coordinate_order(val1, val2)
                        coords.append((lon, lat, alt))
                    except ValueError:
                        continue
        return coords
    
    def format_coordinates(self, coords: List[Tuple[float, float, float]]) -> str:
        """Format coordinates back to KML string"""
        lines = []
        for lon, lat, alt in coords:
            lines.append(f"{lon},{lat},{alt}")
        return '\n'.join(lines)
    
    def convert_coordinates(self, coords: List[Tuple[float, float, float]], 
                          source_crs: str, target_crs: str) -> List[Tuple[float, float, float]]:
        """Convert coordinates between coordinate systems"""
        if not PYPROJ_AVAILABLE or Transformer is None:
            return coords
        
        if source_crs == target_crs:
            return coords
        
        try:
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            converted = []
            for lon, lat, alt in coords:
                new_lon, new_lat = transformer.transform(lon, lat)
                converted.append((new_lon, new_lat, alt))
            return converted
        except Exception as e:
            st.warning(f"CRS conversion error: {str(e)}")
            return coords
    
    def get_all_points(self) -> List[Tuple[float, float]]:
        """Extract all Point coordinates from KML"""
        points = []
        placemarks = self.root.xpath('.//kml:Placemark', namespaces=self.NS)
        
        for placemark in placemarks:
            point_elem = placemark.find('kml:Point', namespaces=self.NS)
            if point_elem is not None:
                coord_elem = point_elem.find('kml:coordinates', namespaces=self.NS)
                if coord_elem is not None and coord_elem.text:
                    coords = self.extract_coordinates(coord_elem.text)
                    if coords:
                        points.append((coords[0][0], coords[0][1]))
        
        return points
    
    def get_all_polygons(self) -> List[Dict]:
        """Extract all Polygon coordinates from KML"""
        polygons = []
        placemarks = self.root.xpath('.//kml:Placemark', namespaces=self.NS)
        
        for placemark in placemarks:
            polygon_elem = placemark.find('.//kml:Polygon', namespaces=self.NS)
            if polygon_elem is not None:
                name_elem = placemark.find('kml:name', namespaces=self.NS)
                name = name_elem.text if name_elem is not None else "Polygon"
                
                outer_boundary = polygon_elem.find('kml:outerBoundaryIs/kml:LinearRing', namespaces=self.NS)
                if outer_boundary is not None:
                    coord_elem = outer_boundary.find('kml:coordinates', namespaces=self.NS)
                    if coord_elem is not None and coord_elem.text:
                        coords = self.extract_coordinates(coord_elem.text)
                        if coords:
                            polygons.append({
                                'name': name,
                                'coords': [(c[0], c[1]) for c in coords]
                            })
        
        return polygons
    
    def find_nearest_point(self, vertex: Tuple[float, float], 
                          points: List[Tuple[float, float]], 
                          tolerance: float = 0.0001) -> Tuple[float, float]:
        """Find nearest point to a polygon vertex within tolerance"""
        if not points:
            return vertex
        
        distances = [np.sqrt((vertex[0] - p[0])**2 + (vertex[1] - p[1])**2) for p in points]
        min_dist = min(distances)
        
        if min_dist <= tolerance:
            nearest_idx = distances.index(min_dist)
            return points[nearest_idx]
        
        return vertex
    
    def create_polygon_with_style(self, points: List[Tuple[float, float]]) -> etree._Element:
        """Create a styled polygon from point markers"""
        polygon_elem = etree.Element('{http://www.opengis.net/kml/2.2}Polygon')
        
        extrude = etree.SubElement(polygon_elem, '{http://www.opengis.net/kml/2.2}extrude')
        extrude.text = '0'
        
        tessellate = etree.SubElement(polygon_elem, '{http://www.opengis.net/kml/2.2}tessellate')
        tessellate.text = '1'
        
        altitudeMode = etree.SubElement(polygon_elem, '{http://www.opengis.net/kml/2.2}altitudeMode')
        altitudeMode.text = 'clampToGround'
        
        outer_boundary = etree.SubElement(polygon_elem, '{http://www.opengis.net/kml/2.2}outerBoundaryIs')
        linear_ring = etree.SubElement(outer_boundary, '{http://www.opengis.net/kml/2.2}LinearRing')
        
        coord_elem = etree.SubElement(linear_ring, '{http://www.opengis.net/kml/2.2}coordinates')
        
        coords = [(p[0], p[1], 0.0) for p in points]
        coords.append(coords[0])
        coord_elem.text = self.format_coordinates(coords)
        
        return polygon_elem
    
    def snap_polygons(self, tolerance: float = 0.0001) -> Tuple[int, bool]:
        """Snap polygon vertices to point markers"""
        points = self.get_all_points()
        if not points:
            return 0, False
        
        snapped_count = 0
        created_polygon = False
        placemarks = self.root.xpath('.//kml:Placemark', namespaces=self.NS)
        
        for placemark in placemarks:
            polygon_elem = placemark.find('.//kml:Polygon', namespaces=self.NS)
            if polygon_elem is not None:
                name_elem = placemark.find('kml:name', namespaces=self.NS)
                if name_elem is None:
                    name_elem = etree.Element('{http://www.opengis.net/kml/2.2}name')
                    placemark.insert(0, name_elem)
                
                if not name_elem.text or name_elem.text.strip() == '':
                    name_elem.text = f"{self.filename} POLYGON"
                
                outer_boundary = polygon_elem.find('kml:outerBoundaryIs/kml:LinearRing', namespaces=self.NS)
                if outer_boundary is not None:
                    coord_elem = outer_boundary.find('kml:coordinates', namespaces=self.NS)
                    if coord_elem is not None and coord_elem.text:
                        coords = self.extract_coordinates(coord_elem.text)
                        snapped_coords = []
                        for lon, lat, alt in coords:
                            new_lon, new_lat = self.find_nearest_point((lon, lat), points, tolerance)
                            if (new_lon, new_lat) != (lon, lat):
                                snapped_count += 1
                            snapped_coords.append((new_lon, new_lat, alt))
                        coord_elem.text = self.format_coordinates(snapped_coords)
                
                inner_boundaries = polygon_elem.findall('kml:innerBoundaryIs/kml:LinearRing', namespaces=self.NS)
                for inner_boundary in inner_boundaries:
                    coord_elem = inner_boundary.find('kml:coordinates', namespaces=self.NS)
                    if coord_elem is not None and coord_elem.text:
                        coords = self.extract_coordinates(coord_elem.text)
                        snapped_coords = []
                        for lon, lat, alt in coords:
                            new_lon, new_lat = self.find_nearest_point((lon, lat), points, tolerance)
                            if (new_lon, new_lat) != (lon, lat):
                                snapped_count += 1
                            snapped_coords.append((new_lon, new_lat, alt))
                        coord_elem.text = self.format_coordinates(snapped_coords)
        
        existing_polygon = self.root.find('.//kml:Polygon', namespaces=self.NS)
        if existing_polygon is None and len(points) >= 3:
            document = self.root.find('.//kml:Document', namespaces=self.NS)
            if document is None:
                document = self.root.find('kml:Document', namespaces=self.NS)
            
            if document is not None:
                placemark = etree.SubElement(document, '{http://www.opengis.net/kml/2.2}Placemark')
                
                name = etree.SubElement(placemark, '{http://www.opengis.net/kml/2.2}name')
                name.text = f"{self.filename} POLYGON"
                
                styleUrl = etree.SubElement(placemark, '{http://www.opengis.net/kml/2.2}styleUrl')
                styleUrl.text = '#polygonStyle'
                
                polygon_elem = self.create_polygon_with_style(points)
                placemark.append(polygon_elem)
                
                self._add_polygon_style(document)
                
                created_polygon = True
                snapped_count = len(points)
        
        return snapped_count, created_polygon
    
    def _add_polygon_style(self, document):
        """Add polygon style to document"""
        existing_style = self.root.find('.//kml:Style[@id="polygonStyle"]', namespaces=self.NS)
        if existing_style is not None:
            return
        
        style = etree.Element('{http://www.opengis.net/kml/2.2}Style')
        style.set('id', 'polygonStyle')
        
        line_style = etree.SubElement(style, '{http://www.opengis.net/kml/2.2}LineStyle')
        line_color = etree.SubElement(line_style, '{http://www.opengis.net/kml/2.2}color')
        line_color.text = 'ff0000ff'
        line_width = etree.SubElement(line_style, '{http://www.opengis.net/kml/2.2}width')
        line_width.text = '3'
        
        poly_style = etree.SubElement(style, '{http://www.opengis.net/kml/2.2}PolyStyle')
        poly_color = etree.SubElement(poly_style, '{http://www.opengis.net/kml/2.2}color')
        poly_color.text = 'ff0000ff'
        poly_fill = etree.SubElement(poly_style, '{http://www.opengis.net/kml/2.2}fill')
        poly_fill.text = '1'
        poly_outline = etree.SubElement(poly_style, '{http://www.opengis.net/kml/2.2}outline')
        poly_outline.text = '1'
        
        document.append(style)
    
    def get_corrected_kml(self) -> str:
        """Get corrected KML as string"""
        return etree.tostring(self.root, pretty_print=True, 
                            xml_declaration=True, encoding='UTF-8').decode('utf-8')
    
    def get_stats(self) -> Dict:
        """Get statistics about the KML"""
        points = self.get_all_points()
        placemarks = self.root.xpath('.//kml:Placemark', namespaces=self.NS)
        polygons = len(self.root.xpath('.//kml:Polygon', namespaces=self.NS))
        
        return {
            'points': len(points),
            'placemarks': len(placemarks),
            'polygons': polygons,
            'has_polygons': polygons > 0
        }


def create_shapefile_from_kml(kml_content: str, filename: str, 
                             source_crs: str, target_crs: str) -> Optional[Dict]:
    """Create shapefile from KML content with optional CRS conversion"""
    if not SHAPEFILE_AVAILABLE or shapefile is None:
        return None
    
    try:
        corrector = KMLCorrector(kml_content, filename)
        points = corrector.get_all_points()
        polygons = corrector.get_all_polygons()
        
        if not points and not polygons:
            return None
        
        # Convert coordinates if needed
        if PYPROJ_AVAILABLE and source_crs != target_crs:
            points = [corrector.convert_coordinates([(p[0], p[1], 0)], source_crs, target_crs)[0][:2] for p in points]
            for poly in polygons:
                converted_coords = corrector.convert_coordinates(
                    [(c[0], c[1], 0) for c in poly['coords']], 
                    source_crs, target_crs
                )
                poly['coords'] = [(c[0], c[1]) for c in converted_coords]
        
        base_filename = Path(filename).stem
        base_filename = "".join(c for c in base_filename if c.isalnum() or c in ('-', '_'))
        
        if not base_filename or len(base_filename.strip()) == 0:
            base_filename = "shapefile"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            shp_path = os.path.join(temp_dir, base_filename)
            
            try:
                # Try POLYGON first if we have polygons
                if polygons and len(polygons) > 0:
                    valid_polys = [p for p in polygons if len(p['coords']) >= 3]
                    
                    if valid_polys:
                        try:
                            w = shapefile.Writer(shp_path, shapeType=shapefile.POLYGON)
                            w.field('name', 'C', '50')
                            w.field('type', 'C', '20')
                            
                            for poly in valid_polys:
                                coords = [(float(lon), float(lat)) for lon, lat in poly['coords']]
                                w.poly([coords])
                                w.record(str(poly['name'])[:50], 'Polygon')
                            
                            w.close()
                            
                            shp_file = f"{shp_path}.shp"
                            shx_file = f"{shp_path}.shx"
                            dbf_file = f"{shp_path}.dbf"
                            
                            if os.path.exists(shp_file) and os.path.exists(shx_file) and os.path.exists(dbf_file):
                                with open(shp_file, 'rb') as f:
                                    shp_data = f.read()
                                with open(shx_file, 'rb') as f:
                                    shx_data = f.read()
                                with open(dbf_file, 'rb') as f:
                                    dbf_data = f.read()
                                
                                if shp_data and shx_data and dbf_data:
                                    return {
                                        'shp': shp_data,
                                        'shx': shx_data,
                                        'dbf': dbf_data,
                                    }
                        except Exception as poly_err:
                            pass
                
                # Fall back to POINT shapefile
                w = shapefile.Writer(shp_path, shapeType=shapefile.POINT)
                w.field('name', 'C', '50')
                w.field('type', 'C', '20')
                
                for i, (lon, lat) in enumerate(points):
                    w.point(float(lon), float(lat))
                    w.record(f"Point_{i+1}", 'Point')
                
                w.close()
                
                shp_file = f"{shp_path}.shp"
                shx_file = f"{shp_path}.shx"
                dbf_file = f"{shp_path}.dbf"
                
                if os.path.exists(shp_file) and os.path.exists(shx_file) and os.path.exists(dbf_file):
                    with open(shp_file, 'rb') as f:
                        shp_data = f.read()
                    with open(shx_file, 'rb') as f:
                        shx_data = f.read()
                    with open(dbf_file, 'rb') as f:
                        dbf_data = f.read()
                    
                    if shp_data and shx_data and dbf_data:
                        return {
                            'shp': shp_data,
                            'shx': shx_data,
                            'dbf': dbf_data,
                        }
                
                return None
                    
            except Exception as inner_err:
                return None
    
    except Exception as e:
        return None


st.title("🗺️ KML Corrector Tool v2.4.1")
st.markdown("**CRS conversion for Shapefile only - KML stays original ✨**")

if not SHAPEFILE_AVAILABLE:
    st.warning("⚠️ Shapefile support not available - Install: `pip install pyshp`")

if not PYPROJ_AVAILABLE:
    st.info("ℹ️ CRS conversion available - Install: `pip install pyproj`")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Features:
    1. **Auto-detect coordinate order** ✅
    2. **Custom precision slider** ✅
    3. **Correct snapping** - Points FIXED, polygon snaps ✅
    4. **Styled polygons** - Red, 3px ✅
    5. **Batch processing** ✅
    6. **Export** - KML + Shapefile ✅
    7. **Folder support** - Finds placemarks in Folders ✅
    8. **CRS for Shapefile only** - KML unchanged ✨
    """)

with col2:
    st.info("⚙️ **Custom Precision**")
    st.write("**Snap Tolerance (degrees):**")
    tolerance = st.slider(
        "Adjust precision level",
        min_value=0.00001,
        max_value=0.1,
        value=0.0001,
        step=0.00001,
        format="%.5f",
        label_visibility="collapsed"
    )
    distance_m = tolerance * 111000
    st.metric("Distance (approx)", f"{distance_m:.1f} m")

st.divider()

# Coordinate System Selection - FOR SHAPEFILE ONLY
st.subheader("🌍 Shapefile Coordinate System (KML stays original)")

col_src, col_tgt = st.columns(2)

with col_src:
    source_crs = st.selectbox(
        "Source CRS (your current coordinates):",
        list(CRS_OPTIONS.keys()),
        index=0,
        help="The coordinate system of your input KML"
    )

with col_tgt:
    target_crs = st.selectbox(
        "Target CRS (for Shapefile export):",
        list(CRS_OPTIONS.keys()),
        index=0,
        help="Convert shapefile to this coordinate system. KML stays unchanged."
    )

st.info("📌 **Note:** KML files are always exported in their original coordinate system. Only Shapefile will be converted.")

st.divider()

with st.expander("📊 Tolerance Reference (Click to expand)", expanded=False):
    st.write("**Quick reference:**")
    tolerance_data = {
        'Tolerance': ['0.00001°', '0.0001°', '0.001°', '0.01°'],
        'Distance': ['1.1 m', '11 m', '111 m', '1.1 km'],
        'Use Case': ['Survey', 'Flight planning', 'GPS', 'Coarse']
    }
    st.table(tolerance_data)

st.divider()

uploaded_files = st.file_uploader(
    "Upload KML files",
    type=['kml'],
    accept_multiple_files=True,
    help="Upload KML files"
)

if uploaded_files:
    st.divider()
    
    results = []
    all_corrected = {}
    all_shapefiles = {}
    shapefile_errors = {}
    
    for uploaded_file in uploaded_files:
        try:
            kml_content = uploaded_file.read().decode('utf-8')
            corrector = KMLCorrector(kml_content, uploaded_file.name)
            
            stats = corrector.get_stats()
            snapped, created_polygon = corrector.snap_polygons(tolerance)
            
            # KML always stays in original coordinate system - NO CONVERSION
            corrected_kml = corrector.get_corrected_kml()
            all_corrected[uploaded_file.name] = corrected_kml
            
            # Shapefile gets CRS conversion
            if SHAPEFILE_AVAILABLE:
                shp_data = create_shapefile_from_kml(
                    kml_content, 
                    uploaded_file.name,
                    CRS_OPTIONS[source_crs],
                    CRS_OPTIONS[target_crs]
                )
                if shp_data:
                    all_shapefiles[uploaded_file.name] = shp_data
                else:
                    shapefile_errors[uploaded_file.name] = "pyshp unavailable or file I/O error"
            
            status_parts = []
            if stats['has_polygons']:
                status_parts.append(f"✅ Snapped {snapped} vertices")
            else:
                status_parts.append("⚠️ No polygons")
            
            if created_polygon:
                status_parts.append("🆕 Created polygon")
            
            results.append({
                'filename': uploaded_file.name,
                'points': stats['points'],
                'polygons': stats['polygons'],
                'snapped': snapped,
                'status': ' | '.join(status_parts)
            })
            
        except Exception as e:
            results.append({
                'filename': uploaded_file.name,
                'points': 0,
                'polygons': 0,
                'snapped': 0,
                'status': f'❌ {str(e)[:40]}'
            })
    
    st.subheader("📋 Processing Results")
    st.dataframe(
        results,
        width='stretch',
        hide_index=True,
        column_config={
            'filename': 'File',
            'points': st.column_config.NumberColumn('Points'),
            'polygons': st.column_config.NumberColumn('Polygons'),
            'snapped': st.column_config.NumberColumn('Snapped'),
            'status': 'Status'
        }
    )
    
    if shapefile_errors:
        st.warning(f"⚠️ Shapefile issues: {len(shapefile_errors)} file(s)")
        for fname, error in shapefile_errors.items():
            st.write(f"  • {fname}: {error}")
    
    st.divider()
    st.subheader("📥 Download")
    
    # Show CRS info
    if PYPROJ_AVAILABLE and source_crs != target_crs:
        st.success(f"🗺️ Shapefile converting: {source_crs} → {target_crs}")
        st.info("📄 KML stays in original coordinate system")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📄 KML (Original CRS)**")
        if len(all_corrected) == 1:
            filename = list(all_corrected.keys())[0]
            st.download_button(
                label="📄 Download KML",
                data=all_corrected[filename],
                file_name=filename,
                mime="application/vnd.google-earth.kml+xml",
                width='stretch'
            )
        else:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fname, content in all_corrected.items():
                    zf.writestr(fname, content)
            zip_buffer.seek(0)
            st.download_button(
                label="📦 All KML (ZIP)",
                data=zip_buffer,
                file_name="corrected_kmls.zip",
                mime="application/zip",
                width='stretch'
            )
        st.info(f"✅ {len(all_corrected)} KML ready (unchanged)")
    
    with col2:
        st.write("**🗺️ Shapefile (Converted CRS)**")
        if SHAPEFILE_AVAILABLE:
            if all_shapefiles:
                if len(all_shapefiles) == 1:
                    filename = list(all_shapefiles.keys())[0]
                    shp_data = all_shapefiles[filename]
                    
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        base_name = Path(filename).stem
                        zf.writestr(f"{base_name}.shp", shp_data['shp'])
                        zf.writestr(f"{base_name}.shx", shp_data['shx'])
                        zf.writestr(f"{base_name}.dbf", shp_data['dbf'])
                    zip_buffer.seek(0)
                    st.download_button(
                        label="🗺️ Download Shapefile",
                        data=zip_buffer,
                        file_name=f"{Path(filename).stem}_shapefile.zip",
                        mime="application/zip",
                        width='stretch'
                    )
                else:
                    combined_zip = BytesIO()
                    with zipfile.ZipFile(combined_zip, 'w', zipfile.ZIP_DEFLATED) as oz:
                        for fname, shp_data in all_shapefiles.items():
                            base_name = Path(fname).stem
                            folder = f"{base_name}/"
                            oz.writestr(f"{folder}{base_name}.shp", shp_data['shp'])
                            oz.writestr(f"{folder}{base_name}.shx", shp_data['shx'])
                            oz.writestr(f"{folder}{base_name}.dbf", shp_data['dbf'])
                    combined_zip.seek(0)
                    st.download_button(
                        label="🗺️ All Shapefiles (ZIP)",
                        data=combined_zip,
                        file_name="all_shapefiles.zip",
                        mime="application/zip",
                        width='stretch'
                    )
                st.info(f"✅ {len(all_shapefiles)} Shapefile(s) ready (converted)")
            else:
                st.info("No shapefiles generated")
        else:
            st.warning("Install pyshp to enable shapefile export")
    
    st.divider()
    st.write(f"**Status:** {len(all_corrected)} KML file(s) | {len(all_shapefiles)} Shapefile(s) | Tolerance: {tolerance}°")
    
    with st.expander("📋 View XML"):
        for filename, content in all_corrected.items():
            with st.expander(f"📄 {filename}"):
                st.code(content, language="xml", line_numbers=True)

else:
    st.info("👆 Upload KML files to start")
    st.markdown("""
    ### How to Use:
    1. Select source coordinate system (your current coordinates)
    2. Select target coordinate system (for Shapefile export only)
    3. Adjust precision slider
    4. Upload KML files
    5. Download:
       - **KML**: Original coordinate system (unchanged)
       - **Shapefile**: Converted to target coordinate system
    """)
