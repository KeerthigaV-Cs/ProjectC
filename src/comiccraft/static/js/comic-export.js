const pdfDownloadLink = document.querySelector("[data-pdf-download]");

if (pdfDownloadLink) {
  const downloadStatus = document.querySelector("#pdf-download-status");
  let downloadInProgress = false;

  pdfDownloadLink.addEventListener("click", async (event) => {
    event.preventDefault();
    if (downloadInProgress) {
      return;
    }

    downloadInProgress = true;
    pdfDownloadLink.setAttribute("aria-busy", "true");

    try {
      const response = await fetch(pdfDownloadLink.href);
      if (!response.ok) {
        throw new Error("PDF download failed.");
      }

      const pdfBlob = await response.blob();
      const objectUrl = URL.createObjectURL(pdfBlob);
      const downloadLink = document.createElement("a");
      downloadLink.href = objectUrl;
      downloadLink.download = pdfDownloadLink.download;
      document.body.append(downloadLink);
      downloadLink.click();
      downloadLink.remove();

      window.setTimeout(() => {
        URL.revokeObjectURL(objectUrl);
        window.location.assign(pdfDownloadLink.dataset.successUrl);
      }, 150);
    } catch {
      downloadInProgress = false;
      pdfDownloadLink.removeAttribute("aria-busy");
      if (downloadStatus) {
        downloadStatus.textContent = "The PDF could not be downloaded. Please try again.";
        downloadStatus.hidden = false;
      }
    }
  });
}