
import { jsPDF } from 'jspdf';

interface ReportSection {
  title: string;
  content: string;
}

// Google Fonts CDN for Roboto-Regular which supports Turkish characters
const FONT_URL = 'https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Me5Q.ttf';

export const generateResearchReport = async (
  reportTitle: string,
  fileName: string,
  sections: ReportSection[]
) => {
  const doc = new jsPDF();
  
  // --- 1. LOAD CUSTOM FONT FOR TURKISH SUPPORT ---
  try {
    const response = await fetch(FONT_URL);
    if (!response.ok) throw new Error("Font yüklenemedi");
    const blob = await response.blob();
    
    const reader = new FileReader();
    await new Promise((resolve) => {
        reader.onloadend = resolve;
        reader.readAsDataURL(blob);
    });

    const base64data = (reader.result as string).split(',')[1];
    
    // Add font to VFS
    doc.addFileToVFS('Roboto-Regular.ttf', base64data);
    doc.addFont('Roboto-Regular.ttf', 'Roboto', 'normal');
    doc.setFont('Roboto'); // Set as default font
  } catch (e) {
    console.warn("Özel font yüklenemedi, varsayılan font kullanılıyor. Türkçe karakterler bozuk görünebilir.", e);
  }

  const pageWidth = doc.internal.pageSize.width;
  const pageHeight = doc.internal.pageSize.height;
  const margin = 20;

  // --- Colors ---
  const tealDark = '#0f2d33';
  const tealLight = '#2dd4bf';
  const slateGray = '#475569';

  let currentPage = 1;

  // Helper: Header
  const addHeader = () => {
    doc.setFillColor(tealDark);
    doc.rect(0, 0, pageWidth, 15, 'F'); // Top bar
    doc.setFontSize(10);
    doc.setTextColor(255, 255, 255);
    doc.text('ScholarSphere Nexus™ 2.0 | Akademik Araştırma Raporu', margin, 10);
  };

  // Helper: Footer
  const addFooter = (pageNo: number) => {
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    const text = `Sayfa ${pageNo}`;
    const textWidth = doc.getTextWidth(text);
    doc.text(text, pageWidth - margin - textWidth, pageHeight - 10);
    doc.text(`Oluşturulma Tarihi: ${new Date().toLocaleDateString('tr-TR')}`, margin, pageHeight - 10);
  };

  // --- COVER PAGE ---
  doc.setFillColor(tealDark);
  doc.rect(0, 0, pageWidth, pageHeight, 'F'); // Full background

  // Logo/Icon placeholder (Text based)
  doc.setFontSize(40);
  doc.setTextColor(tealLight);
  doc.text('ScholarSphere', pageWidth / 2, 100, { align: 'center' });
  doc.setFontSize(20);
  doc.setTextColor(255, 255, 255);
  doc.text('NEXUS 2.0', pageWidth / 2, 115, { align: 'center' });

  // Report Title
  doc.setFontSize(24); // Slightly smaller to fit long Turkish titles
  doc.setTextColor(255, 255, 255);
  
  // Split title if too long
  const splitTitle = doc.splitTextToSize(reportTitle, pageWidth - 40);
  doc.text(splitTitle, pageWidth / 2, 160, { align: 'center' });

  // Sub Info
  doc.setFontSize(14);
  doc.setTextColor(200, 200, 200);
  doc.text(`İncelenen Kaynak: ${fileName}`, pageWidth / 2, 190, { align: 'center' });
  doc.text(new Date().toLocaleDateString('tr-TR', { year: 'numeric', month: 'long', day: 'numeric' }), pageWidth / 2, 200, { align: 'center' });

  // Footer on Cover
  doc.setFontSize(10);
  doc.text('Profesyonel Akademik Çıktı', pageWidth / 2, pageHeight - 30, { align: 'center' });

  // --- CONTENT PAGES ---
  doc.addPage();
  addHeader();
  addFooter(2);
  currentPage = 2;

  let yPos = 30;

  sections.forEach((section) => {
    // Section Title
    if (yPos > pageHeight - 40) {
      doc.addPage();
      addHeader();
      currentPage++;
      addFooter(currentPage);
      yPos = 30;
    }

    doc.setFontSize(16);
    doc.setTextColor(tealDark);
    // Note: 'bold' style might require Roboto-Bold.ttf, sticking to normal to be safe or mapping it if loaded
    // Using normal weight but larger size for title to keep file size small (only 1 font file)
    doc.setFont('Roboto', 'normal'); 
    doc.text(section.title, margin, yPos);
    yPos += 10;

    // Draw Line under title
    doc.setDrawColor(tealLight);
    doc.setLineWidth(0.5);
    doc.line(margin, yPos - 5, pageWidth - margin, yPos - 5);

    // Content
    doc.setFontSize(11);
    doc.setTextColor(slateGray);

    // Split text to wrap
    const splitText = doc.splitTextToSize(section.content, pageWidth - (margin * 2));
    
    // Check if text fits
    const textHeight = splitText.length * 7; // approx line height
    if (yPos + textHeight > pageHeight - 30) {
       // It doesn't fit, print line by line or chunks
       splitText.forEach((line: string) => {
          if (yPos > pageHeight - 30) {
             doc.addPage();
             addHeader();
             currentPage++;
             addFooter(currentPage);
             yPos = 30;
          }
          doc.text(line, margin, yPos);
          yPos += 7;
       });
    } else {
       doc.text(splitText, margin, yPos);
       yPos += textHeight + 10; // Add spacing after section
    }
    
    yPos += 5; // Extra padding
  });

  // Download
  doc.save(`ScholarSphere_Report_${Date.now()}.pdf`);
};
